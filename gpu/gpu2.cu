#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "../common/common.hpp"
#include "../common/solver.hpp"
#include <stdio.h>

// Structure to hold simulation parameters
struct SimParams {
    int nx, ny;
    double H, g;
    double dx, dy, dt;
    int t;
};

// Global device memory pointers
struct DeviceArrays {
    // Current fields
    double *d_h, *d_u, *d_v;
    // Current derivatives
    double *d_dh, *d_du, *d_dv;
    // Previous derivatives
    double *d_dh1, *d_du1, *d_dv1;
    // Two steps ago derivatives
    double *d_dh2, *d_du2, *d_dv2;
};

// Global instances
SimParams params = {0};
DeviceArrays d_arrays = {0};

__global__ void compute_boundaries(double *h, double *u, double *v, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < ny) {
        // Periodic boundary in x-direction
        h[nx + idx*(nx+1)] = h[idx*(nx+1)];      // h(nx,i) = h(0,i)
        u[idx*nx] = u[nx + idx*nx];              // u(0,i) = u(nx,i)
    }
    else if (idx < nx + ny) {
        int i = idx - ny;
        // Periodic boundary in y-direction
        h[i + ny*(nx+1)] = h[i];                 // h(i,ny) = h(i,0)
        v[i] = v[i + ny*nx];                     // v(i,0) = v(i,ny)
    }
}

__global__ void compute_derivatives_and_update(
    double *h, double *u, double *v,
    double *dh, double *du, double *dv,
    double *dh1, double *du1, double *dv1,
    double *dh2, double *du2, double *dv2,
    const SimParams params
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= params.nx || j >= params.ny) return;

    // Compute derivatives
    dh[i + j*params.nx] = -params.H * (
        (u[i+1 + j*params.nx] - u[i + j*params.nx]) / params.dx +
        (v[i + (j+1)*params.nx] - v[i + j*params.nx]) / params.dy
    );
    
    du[i + j*params.nx] = -params.g * 
        (h[i+1 + j*(params.nx+1)] - h[i + j*(params.nx+1)]) / params.dx;
    
    dv[i + j*params.nx] = -params.g * 
        (h[i + (j+1)*(params.nx+1)] - h[i + j*(params.nx+1)]) / params.dy;

    __syncthreads();

    // Calculate multistep coefficients
    double a1, a2 = 0.0, a3 = 0.0;
    if (params.t == 0) {
        a1 = 1.0;
    } else if (params.t == 1) {
        a1 = 3.0/2.0;
        a2 = -1.0/2.0;
    } else {
        a1 = 23.0/12.0;
        a2 = -16.0/12.0;
        a3 = 5.0/12.0;
    }

    // Update fields
    h[i + j*(params.nx+1)] += (
        a1 * dh[i + j*params.nx] +
        a2 * dh1[i + j*params.nx] +
        a3 * dh2[i + j*params.nx]
    ) * params.dt;
    
    u[i+1 + j*params.nx] += (
        a1 * du[i + j*params.nx] +
        a2 * du1[i + j*params.nx] +
        a3 * du2[i + j*params.nx]
    ) * params.dt;
    
    v[i + (j+1)*params.nx] += (
        a1 * dv[i + j*params.nx] +
        a2 * dv1[i + j*params.nx] +
        a3 * dv2[i + j*params.nx]
    ) * params.dt;
}

void init(double *h0, double *u0, double *v0, double length, double width,
          int nx_, int ny_, double H_, double g_, double dt_, int rank, int num_procs) {
    // Initialize parameters
    params.nx = nx_;
    params.ny = ny_;
    params.H = H_;
    params.g = g_;
    params.dx = length / nx_;
    params.dy = width / ny_;
    params.dt = dt_;
    params.t = 0;
    
    // Allocate device memory
    size_t field_size_h = (params.nx + 1) * (params.ny + 1) * sizeof(double);
    size_t field_size_u = (params.nx + 2) * params.ny * sizeof(double);
    size_t field_size_v = params.nx * (params.ny + 2) * sizeof(double);
    size_t deriv_size = params.nx * params.ny * sizeof(double);
    
    // Allocate fields
    cudaMalloc(&d_arrays.d_h, field_size_h);
    cudaMalloc(&d_arrays.d_u, field_size_u);
    cudaMalloc(&d_arrays.d_v, field_size_v);
    
    // Allocate derivatives
    cudaMalloc(&d_arrays.d_dh, deriv_size);
    cudaMalloc(&d_arrays.d_du, deriv_size);
    cudaMalloc(&d_arrays.d_dv, deriv_size);
    cudaMalloc(&d_arrays.d_dh1, deriv_size);
    cudaMalloc(&d_arrays.d_du1, deriv_size);
    cudaMalloc(&d_arrays.d_dv1, deriv_size);
    cudaMalloc(&d_arrays.d_dh2, deriv_size);
    cudaMalloc(&d_arrays.d_du2, deriv_size);
    cudaMalloc(&d_arrays.d_dv2, deriv_size);
    
    // Copy initial conditions
    cudaMemcpy(d_arrays.d_h, h0, field_size_h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arrays.d_u, u0, field_size_u, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arrays.d_v, v0, field_size_v, cudaMemcpyHostToDevice);
    
    // Initialize derivatives to zero
    cudaMemset(d_arrays.d_dh, 0, deriv_size);
    cudaMemset(d_arrays.d_du, 0, deriv_size);
    cudaMemset(d_arrays.d_dv, 0, deriv_size);
    cudaMemset(d_arrays.d_dh1, 0, deriv_size);
    cudaMemset(d_arrays.d_du1, 0, deriv_size);
    cudaMemset(d_arrays.d_dv1, 0, deriv_size);
    cudaMemset(d_arrays.d_dh2, 0, deriv_size);
    cudaMemset(d_arrays.d_du2, 0, deriv_size);
    cudaMemset(d_arrays.d_dv2, 0, deriv_size);
}

void step() {
    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((params.nx + block.x - 1) / block.x, 
              (params.ny + block.y - 1) / block.y);

    dim3 boundary_block(256);
    dim3 boundary_grid((params.nx + params.ny + boundary_block.x - 1) / 
                       boundary_block.x);

    // Update boundaries
    compute_boundaries<<<boundary_grid, boundary_block>>>(
        d_arrays.d_h, d_arrays.d_u, d_arrays.d_v, 
        params.nx, params.ny
    );
    
    // Compute derivatives and update fields
    compute_derivatives_and_update<<<grid, block>>>(
        d_arrays.d_h, d_arrays.d_u, d_arrays.d_v,
        d_arrays.d_dh, d_arrays.d_du, d_arrays.d_dv,
        d_arrays.d_dh1, d_arrays.d_du1, d_arrays.d_dv1,
        d_arrays.d_dh2, d_arrays.d_du2, d_arrays.d_dv2,
        params
    );
    
    // Swap derivative buffers
    double *tmp;
    tmp = d_arrays.d_dh2; d_arrays.d_dh2 = d_arrays.d_dh1; 
    d_arrays.d_dh1 = d_arrays.d_dh; d_arrays.d_dh = tmp;
    
    tmp = d_arrays.d_du2; d_arrays.d_du2 = d_arrays.d_du1; 
    d_arrays.d_du1 = d_arrays.d_du; d_arrays.d_du = tmp;
    
    tmp = d_arrays.d_dv2; d_arrays.d_dv2 = d_arrays.d_dv1; 
    d_arrays.d_dv1 = d_arrays.d_dv; d_arrays.d_dv = tmp;
    
    params.t++;
}

void transfer(double *h_host) {
    cudaMemcpy(h_host, d_arrays.d_h, 
               (params.nx + 1) * (params.ny + 1) * sizeof(double), 
               cudaMemcpyDeviceToHost);
}

void free_memory() {
    cudaFree(d_arrays.d_h);
    cudaFree(d_arrays.d_u);
    cudaFree(d_arrays.d_v);
    cudaFree(d_arrays.d_dh);
    cudaFree(d_arrays.d_du);
    cudaFree(d_arrays.d_dv);
    cudaFree(d_arrays.d_dh1);
    cudaFree(d_arrays.d_du1);
    cudaFree(d_arrays.d_dv1);
    cudaFree(d_arrays.d_dh2);
    cudaFree(d_arrays.d_du2);
    cudaFree(d_arrays.d_dv2);
}