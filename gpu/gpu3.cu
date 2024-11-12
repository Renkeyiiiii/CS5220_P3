#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "../common/common.hpp"
#include "../common/solver.hpp"

// Global variables for device memory
double *d_h, *d_u, *d_v;           // Current fields
double *d_dh, *d_du, *d_dv;        // Current derivatives
double *d_dh1, *d_du1, *d_dv1;     // Previous derivatives
double *d_dh2, *d_du2, *d_dv2;     // Two steps ago derivatives

// Constants for the simulation
int nx, ny;
int t = 0;

// Constant memory for simulation parameters
__constant__ double d_H;
__constant__ double d_g;
__constant__ double d_dx;
__constant__ double d_dy;
__constant__ double d_dt;

// Combined kernel for all derivative calculations
__global__ void compute_derivatives_kernel(double *h, double *u, double *v,
                                         double *dh, double *du, double *dv,
                                         int nx, int ny) {
    __shared__ double sh_h[34][34];  // Block size + halo regions
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1;  // Local thread index with halo offset
    int ty = threadIdx.y + 1;
    
    // Load central points
    if (i < nx && j < ny) {
        sh_h[ty][tx] = h[i * (ny + 1) + j];
    }
    
    // Load halo regions
    if (threadIdx.x == 0 && i > 0) {
        sh_h[ty][0] = h[(i-1) * (ny + 1) + j];
    }
    if (threadIdx.x == blockDim.x-1 && i < nx-1) {
        sh_h[ty][tx+1] = h[(i+1) * (ny + 1) + j];
    }
    if (threadIdx.y == 0 && j > 0) {
        sh_h[0][tx] = h[i * (ny + 1) + (j-1)];
    }
    if (threadIdx.y == blockDim.y-1 && j < ny-1) {
        sh_h[ty+1][tx] = h[i * (ny + 1) + (j+1)];
    }
    
    __syncthreads();
    
    if (i < nx && j < ny) {
        // Calculate derivatives using shared memory
        double dh_dx = (sh_h[ty][tx+1] - sh_h[ty][tx]) / d_dx;
        double dh_dy = (sh_h[ty+1][tx] - sh_h[ty][tx]) / d_dy;
        
        // Calculate derivatives for velocity
        double du_dx = (u[(i+1) * ny + j] - u[i * ny + j]) / d_dx;
        double dv_dy = (v[i * (ny + 1) + (j+1)] - v[i * (ny + 1) + j]) / d_dy;
        
        // Calculate all derivatives
        int idx = i * ny + j;
        dh[idx] = -d_H * (du_dx + dv_dy);
        du[idx] = -d_g * dh_dx;
        dv[idx] = -d_g * dh_dy;
    }
}

// Kernel for multistep method
__global__ void multistep_kernel(double *h, double *u, double *v,
                                double *dh, double *du, double *dv,
                                double *dh1, double *du1, double *dv1,
                                double *dh2, double *du2, double *dv2,
                                int nx, int ny,
                                double a1, double a2, double a3) {
    __shared__ double sh_dh[32][32];
    __shared__ double sh_du[32][32];
    __shared__ double sh_dv[32][32];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (i < nx && j < ny) {
        int idx = i * ny + j;
        sh_dh[ty][tx] = dh[idx];
        sh_du[ty][tx] = du[idx];
        sh_dv[ty][tx] = dv[idx];
        
        __syncthreads();
        
        // Update height field
        h[i * (ny + 1) + j] += (a1 * sh_dh[ty][tx] + 
                               a2 * dh1[idx] + 
                               a3 * dh2[idx]) * d_dt;
        
        // Update velocity fields
        if (i < nx-1) {
            u[(i+1) * ny + j] += (a1 * sh_du[ty][tx] + 
                                 a2 * du1[idx] + 
                                 a3 * du2[idx]) * d_dt;
        }
        if (j < ny-1) {
            v[i * (ny + 1) + (j+1)] += (a1 * sh_dv[ty][tx] + 
                                       a2 * dv1[idx] + 
                                       a3 * dv2[idx]) * d_dt;
        }
    }
}

// Kernel for boundary conditions
__global__ void compute_boundaries_kernel(double *h, double *u, double *v, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < ny) {
        // Horizontal boundaries
        h[nx * (ny + 1) + idx] = h[idx];
        u[idx] = u[nx * ny + idx];
    }
    
    if (idx < nx) {
        // Vertical boundaries
        h[idx * (ny + 1) + ny] = h[idx * (ny + 1)];
        v[idx * (ny + 1)] = v[idx * (ny + 1) + ny];
    }
}

// CUDA Streams for asynchronous operations
cudaStream_t compute_stream;
cudaStream_t copy_stream;

void init(double *h0, double *u0, double *v0, double length_, double width_, 
          int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_) {
    nx = nx_;
    ny = ny_;
    
    // Copy constants to constant memory
    cudaMemcpyToSymbol(d_H, &H_, sizeof(double));
    cudaMemcpyToSymbol(d_g, &g_, sizeof(double));
    double dx = length_ / nx;
    double dy = width_ / ny;
    cudaMemcpyToSymbol(d_dx, &dx, sizeof(double));
    cudaMemcpyToSymbol(d_dy, &dy, sizeof(double));
    cudaMemcpyToSymbol(d_dt, &dt_, sizeof(double));
    
    // Create CUDA streams
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&copy_stream);
    
    // Allocate device memory
    cudaMalloc(&d_h, (nx + 1) * (ny + 1) * sizeof(double));
    cudaMalloc(&d_u, (nx + 2) * ny * sizeof(double));
    cudaMalloc(&d_v, nx * (ny + 2) * sizeof(double));
    
    size_t deriv_size = nx * ny * sizeof(double);
    cudaMalloc(&d_dh, deriv_size);
    cudaMalloc(&d_du, deriv_size);
    cudaMalloc(&d_dv, deriv_size);
    cudaMalloc(&d_dh1, deriv_size);
    cudaMalloc(&d_du1, deriv_size);
    cudaMalloc(&d_dv1, deriv_size);
    cudaMalloc(&d_dh2, deriv_size);
    cudaMalloc(&d_du2, deriv_size);
    cudaMalloc(&d_dv2, deriv_size);
    
    // Use pinned memory for faster transfers
    double *h0_pinned, *u0_pinned, *v0_pinned;
    cudaMallocHost(&h0_pinned, (nx + 1) * (ny + 1) * sizeof(double));
    cudaMallocHost(&u0_pinned, (nx + 2) * ny * sizeof(double));
    cudaMallocHost(&v0_pinned, nx * (ny + 2) * sizeof(double));
    
    // Copy to pinned memory
    memcpy(h0_pinned, h0, (nx + 1) * (ny + 1) * sizeof(double));
    memcpy(u0_pinned, u0, (nx + 2) * ny * sizeof(double));
    memcpy(v0_pinned, v0, nx * (ny + 2) * sizeof(double));
    
    // Asynchronous copy to device
    cudaMemcpyAsync(d_h, h0_pinned, (nx + 1) * (ny + 1) * sizeof(double), 
                    cudaMemcpyHostToDevice, compute_stream);
    cudaMemcpyAsync(d_u, u0_pinned, (nx + 2) * ny * sizeof(double), 
                    cudaMemcpyHostToDevice, compute_stream);
    cudaMemcpyAsync(d_v, v0_pinned, nx * (ny + 2) * sizeof(double), 
                    cudaMemcpyHostToDevice, compute_stream);
    
    // Initialize derivative arrays
    cudaMemsetAsync(d_dh, 0, deriv_size, compute_stream);
    cudaMemsetAsync(d_du, 0, deriv_size, compute_stream);
    cudaMemsetAsync(d_dv, 0, deriv_size, compute_stream);
    cudaMemsetAsync(d_dh1, 0, deriv_size, compute_stream);
    cudaMemsetAsync(d_du1, 0, deriv_size, compute_stream);
    cudaMemsetAsync(d_dv1, 0, deriv_size, compute_stream);
    cudaMemsetAsync(d_dh2, 0, deriv_size, compute_stream);
    cudaMemsetAsync(d_du2, 0, deriv_size, compute_stream);
    cudaMemsetAsync(d_dv2, 0, deriv_size, compute_stream);
    
    // Free pinned memory
    cudaFreeHost(h0_pinned);
    cudaFreeHost(u0_pinned);
    cudaFreeHost(v0_pinned);
    
    cudaStreamSynchronize(compute_stream);
}

void step() {
    dim3 block(32, 32);  
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    
    // Compute derivatives
    compute_derivatives_kernel<<<grid, block, 0, compute_stream>>>(
        d_h, d_u, d_v, d_dh, d_du, d_dv, nx, ny);
    
    // Set multistep coefficients
    double a1, a2 = 0.0, a3 = 0.0;
    if (t == 0) {
        a1 = 1.0;
    } else if (t == 1) {
        a1 = 3.0 / 2.0;
        a2 = -1.0 / 2.0;
    } else {
        a1 = 23.0 / 12.0;
        a2 = -16.0 / 12.0;
        a3 = 5.0 / 12.0;
    }
    
    // Update fields
    multistep_kernel<<<grid, block, 0, compute_stream>>>(
        d_h, d_u, d_v, d_dh, d_du, d_dv,
        d_dh1, d_du1, d_dv1, d_dh2, d_du2, d_dv2,
        nx, ny, a1, a2, a3);
    
    // Handle boundaries
    dim3 boundary_block(256);
    dim3 boundary_grid((max(nx, ny) + boundary_block.x - 1) / boundary_block.x);
    compute_boundaries_kernel<<<boundary_grid, boundary_block, 0, compute_stream>>>(
        d_h, d_u, d_v, nx, ny);
    
    // Swap derivative buffers
    double *tmp;
    tmp = d_dh2; d_dh2 = d_dh1; d_dh1 = d_dh; d_dh = tmp;
    tmp = d_du2; d_du2 = d_du1; d_du1 = d_du; d_du = tmp;
    tmp = d_dv2; d_dv2 = d_dv1; d_dv1 = d_dv; d_dv = tmp;
    
    cudaStreamSynchronize(compute_stream);
    
    t++;
}

void transfer(double *h_host) {
    // Use pinned memory for faster transfers
    double *h_pinned;
    cudaMallocHost(&h_pinned, (nx + 1) * (ny + 1) * sizeof(double));
    
    // Asynchronous copy from device
    cudaMemcpyAsync(h_pinned, d_h, (nx + 1) * (ny + 1) * sizeof(double), 
                    cudaMemcpyDeviceToHost, copy_stream);
    
    // Synchronize copy stream
    cudaStreamSynchronize(copy_stream);
    
    // Copy to host memory
    memcpy(h_host, h_pinned, (nx + 1) * (ny + 1) * sizeof(double));
    
    // Free pinned memory
    cudaFreeHost(h_pinned);
}

void free_memory() {
    // Free device memory
    cudaFree(d_h);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_dh);
    cudaFree(d_du);
    cudaFree(d_dv);
    cudaFree(d_dh1);
    cudaFree(d_du1);
    cudaFree(d_dv1);
    cudaFree(d_dh2);
    cudaFree(d_du2);
    cudaFree(d_dv2);
    
    // Destroy CUDA streams
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(copy_stream);
    
    // Reset device for clean exit
    cudaDeviceReset();
}