#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "../common/common.hpp"
#include "../common/solver.hpp"
#include <stdio.h>


// Global variables for device memory
double *d_h, *d_u, *d_v;           // Current fields
double *d_dh, *d_du, *d_dv;        // Current derivatives
double *d_dh1, *d_du1, *d_dv1;     // Previous derivatives
double *d_dh2, *d_du2, *d_dv2;     // Two steps ago derivatives

// Constants for the simulation
int nx, ny;
double H, g, dx, dy, dt;
int t = 0;

__global__ void compute_ghost_kernel(double *h, int nx, int ny)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x * gridDim.x;
    int i = index;

    if (index < ny)
    {

            h(nx, i) = h(0, i);

    }
    else if (index < nx + ny)
    {

            h(i-ny, ny) = h(i-ny, 0);

    }
}

__global__ void compute_dh_kernel(double *h, double *u, double *v, double *dh, 
                                int nx, int ny, double H, double dx, double dy) {

    int index_col = blockIdx.x * blockDim.x + threadIdx.x;
    int index_row = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_col = blockDim.x * gridDim.x;
    int stride_row = blockDim.y * gridDim.y;
    for (int i = index_col; i < nx; i += stride_col)
    {
        for (int j = index_row; j < ny; j += stride_row)
        {
            dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
        }
    }
}

__global__ void compute_du_kernel(double *h, double *du, 
                                int nx, int ny, double g, double dx) {

    int index_col = blockIdx.x * blockDim.x + threadIdx.x;
    int index_row = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_col = blockDim.x * gridDim.x;
    int stride_row = blockDim.y * gridDim.y;
    for (int i = index_col; i < nx; i += stride_col)
    {
        for (int j = index_row; j < ny; j += stride_row)
        {
            du(i, j) = -g * dh_dx(i, j);
        }
    }
}

__global__ void compute_dv_kernel(double *h, double *dv, 
                                int nx, int ny, double g, double dy) {

    int index_col = blockIdx.x * blockDim.x + threadIdx.x;
    int index_row = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_col = blockDim.x * gridDim.x;
    int stride_row = blockDim.y * gridDim.y;
    for (int i = index_col; i < nx; i += stride_col)
    {
        for (int j = index_row; j < ny; j += stride_row)
        {
            dv(i, j) = -g * dh_dy(i, j);
        }
    }
}

__global__ void multistep_kernel(double *h, double *u, double *v,
                                double *dh, double *du, double *dv,
                                double *dh1, double *du1, double *dv1,
                                double *dh2, double *du2, double *dv2,
                                int nx, int ny, double dt,
                                double a1, double a2, double a3) 
{

    int index_col = blockIdx.x * blockDim.x + threadIdx.x;
    int index_row = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_col = blockDim.x * gridDim.x;
    int stride_row = blockDim.y * gridDim.y;
    
    for (int i = index_col; i < nx; i += stride_col)
    {
        for (int j = index_row; j < ny; j += stride_row)
        {
            h(i, j) += (a1 * dh(i, j) + a2 * dh1(i, j) + a3 * dh2(i, j)) * dt;
            u(i + 1, j) += (a1 * du(i, j) + a2 * du1(i, j) + a3 * du2(i, j)) * dt;
            v(i, j + 1) += (a1 * dv(i, j) + a2 * dv1(i, j) + a3 * dv2(i, j)) * dt;
        }
    }
}


// We are going to be doing four things here:
// 1.compute_dh
// 2.compute_du
// 3.compute_dv
// 4.multistep
__global__ void new_kernel(double *h, double *u, double *v,
                                double *dh, double *du, double *dv,
                                double *dh1, double *du1, double *dv1,
                                double *dh2, double *du2, double *dv2,
                                int nx, int ny, double dt,
                                double H, double g, double dx, double dy,
                                int t) 
{ 
    int index_col = blockIdx.x * blockDim.x + threadIdx.x;
    int index_row = blockIdx.y * blockDim.y + threadIdx.y;
    int i = index_col;
    int j = index_row;

    if(i < nx && j < ny)
    {
        dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
        du(i, j) = -g * dh_dx(i, j);
        dv(i, j) = -g * dh_dy(i, j);
    }

    __syncthreads();
    // test
    if (t == 100 && i == 0 & j == 0)
    {
        printf("dh(0,0) = %lf\n", dh(0,0));
        printf("du(0,0) = %lf\n", du(0,0));
        printf("dv(0,0) = %lf\n", dv(0,0));
        printf("du_dx(0,0) = %lf\n", du_dx(0,0));
        printf("dv_dy(0,0) = %lf\n", dv_dy(0,0));
        printf("dh_dx(0,0) = %lf\n", dh_dx(0,0));
        printf("dh_dy(0,0) = %lf\n", dh_dy(0,0));
        printf("h(0,0) = %lf h(1,0) = %lf h(0,1) = %lf\n", h(0,0), h(1,0), h(0,1));
        printf("u(0,0) = %lf u(1,0) = %lf\n", u(0,0), u(1,0));
        printf("v(0,0) = %lf v(0,1) = %lf\n", v(0,0), v(0,1));
    }
    __syncthreads();
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

    if(i < nx && j < ny)
    {
        h(i, j) += (a1 * dh(i, j) + a2 * dh1(i, j) + a3 * dh2(i, j)) * dt;
        u(i + 1, j) += (a1 * du(i, j) + a2 * du1(i, j) + a3 * du2(i, j)) * dt;
        v(i, j + 1) += (a1 * dv(i, j) + a2 * dv1(i, j) + a3 * dv2(i, j)) * dt;
    }
}

__global__ void compute_boundaries_kernel(double *h, double *u, double *v, int nx, int ny) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x * gridDim.x;
    int i = index;

    if (index < ny)
    {

            u(0, i) = u(nx, i);

    }
    else if (index < nx + ny)
    {

            v(i-ny, 0) = v(i-ny, ny);

    }
}

void init(double *h0, double *u0, double *v0, double length_, double width_, 
          int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_) {
    // Store grid dimensions and parameters
    nx = nx_;
    ny = ny_;
    H = H_;
    g = g_;
    dx = length_ / nx;
    dy = width_ / ny;
    dt = dt_;
    
    // Allocate device memory with correct sizes
    cudaMalloc(&d_h, (nx + 1) * (ny + 1) * sizeof(double));  // h has extra points in both directions
    cudaMalloc(&d_u, (nx + 2) * ny * sizeof(double));        // u has extra points in x direction
    cudaMalloc(&d_v, nx * (ny + 2) * sizeof(double));        // v has extra points in y direction
    
    // Allocate device memory for derivatives
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
    
    // Copy initial conditions to device with correct sizes
    cudaMemcpy(d_h, h0, (nx + 1) * (ny + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u0, (nx + 2) * ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v0, nx * (ny + 2) * sizeof(double), cudaMemcpyHostToDevice);
    
    // Initialize derivative arrays to zero
    cudaMemset(d_dh, 0, deriv_size);
    cudaMemset(d_du, 0, deriv_size);
    cudaMemset(d_dv, 0, deriv_size);
    cudaMemset(d_dh1, 0, deriv_size);
    cudaMemset(d_du1, 0, deriv_size);
    cudaMemset(d_dv1, 0, deriv_size);
    cudaMemset(d_dh2, 0, deriv_size);
    cudaMemset(d_du2, 0, deriv_size);
    cudaMemset(d_dv2, 0, deriv_size);
}

void step() {
    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    dim3 boundary_block(256);
    dim3 boundary_grid(((nx + ny) + boundary_block.x - 1) / boundary_block.x);

    compute_ghost_kernel<<<boundary_grid, boundary_block>>>(d_h, nx, ny);
    cudaDeviceSynchronize();
    // Compute derivatives

    // compute_dh_kernel<<<grid, block>>>(d_h, d_u, d_v, d_dh, nx, ny, H, dx, dy);
    // cudaDeviceSynchronize();
    // compute_du_kernel<<<grid, block>>>(d_h, d_du, nx, ny, g, dx);
    // cudaDeviceSynchronize();
    // compute_dv_kernel<<<grid, block>>>(d_h, d_dv, nx, ny, g, dy);
    // cudaDeviceSynchronize();

    new_kernel<<<grid, block>>>(d_h, d_u, d_v, d_dh, d_du, d_dv,
                        d_dh1, d_du1, d_dv1, d_dh2, d_du2, d_dv2,
                        nx, ny, dt, H, g, dx, dy, t);
    
    // Set multistep coefficients
    // double a1, a2 = 0.0, a3 = 0.0;
    // if (t == 0) {
    //     a1 = 1.0;
    // } else if (t == 1) {
    //     a1 = 3.0 / 2.0;
    //     a2 = -1.0 / 2.0;
    // } else {
    //     a1 = 23.0 / 12.0;
    //     a2 = -16.0 / 12.0;
    //     a3 = 5.0 / 12.0;
    // }
    
    // Update fields
    // multistep_kernel<<<grid, block>>>(d_h, d_u, d_v, d_dh, d_du, d_dv,
    //                                  d_dh1, d_du1, d_dv1, d_dh2, d_du2, d_dv2,
    //                                  nx, ny, dt, a1, a2, a3);
    // cudaDeviceSynchronize();
    // Handle boundaries
    compute_boundaries_kernel<<<boundary_grid, boundary_block>>>(d_h, d_u, d_v, nx, ny);

    cudaDeviceSynchronize();
    
    // Swap derivative buffers
    double *tmp;
    tmp = d_dh2; d_dh2 = d_dh1; d_dh1 = d_dh; d_dh = tmp;
    tmp = d_du2; d_du2 = d_du1; d_du1 = d_du; d_du = tmp;
    tmp = d_dv2; d_dv2 = d_dv1; d_dv1 = d_dv; d_dv = tmp;
    
    t++;
}

void transfer(double *h_host) {
    cudaMemcpy(h_host, d_h, (nx + 1) * (ny + 1) * sizeof(double), cudaMemcpyDeviceToHost);
}

void free_memory() {
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
}