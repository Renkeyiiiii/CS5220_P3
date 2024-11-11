#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "../common/common.hpp"
#include "../common/solver.hpp"

/*
 * PARALLELIZATION STRATEGY: CPU TO GPU TRANSLATION
 * ==============================================
 * 
 * Original CPU Implementation Structure:
 * 1. Global arrays for fields (h, u, v) and their derivatives
 * 2. Sequential loops for computing derivatives
 * 3. Time integration using multistep method
 * 4. Boundary condition handling
 * 5. Buffer swapping for time evolution
 *
 * GPU Implementation Changes:
 * 1. Replace global arrays with device memory
 * 2. Convert loops to parallel CUDA kernels
 * 3. Use thread/block structure for parallelization
 * 4. Minimize CPU-GPU data transfers
 * 5. Combine related operations into single kernels where possible
 */

// Original CPU code had regular pointers. For GPU, we prefix with d_ to indicate device memory
// and keep them global so all functions can access them
double *d_h, *d_u, *d_v;           // Current fields
double *d_dh, *d_du, *d_dv;        // Current derivatives
double *d_dh1, *d_du1, *d_dv1;     // Previous derivatives
double *d_dh2, *d_du2, *d_dv2;     // Two steps ago derivatives

// Constants remain the same as CPU version
int nx, ny;
double H, g, dx, dy, dt;
int t = 0;

/*
 * CPU TO GPU KERNEL TRANSLATION - Height Derivative
 * Original CPU code:
 * void compute_dh() {
 *     for (int i = 0; i < nx; i++) {
 *         for (int j = 0; j < ny; j++) {
 *             dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
 *         }
 *     }
 * }
 *
 * Changes made for GPU:
 * 1. Add __global__ for CUDA kernel
 * 2. Replace loops with thread/block indices
 * 3. Add bounds checking for thread safety
 * 4. Make data dependencies explicit in parameters
 * 5. Inline derivative calculations instead of using helper functions
 */
__global__ void compute_dh_kernel(double *d_h, double *d_u, double *d_v, double *d_dh, 
                                int nx, int ny, double H, double dx, double dy) {
    // Calculate thread indices - each thread handles one grid point
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Bounds check - ensure thread isn't out of grid bounds
    if (i < nx && j < ny) {
        // Inline the derivative calculations that were separate functions in CPU code
        double du_dx = (d_u[IDX(i+1,j)] - d_u[IDX(i,j)]) / dx;
        double dv_dy = (d_v[IDX(i,j+1)] - d_v[IDX(i,j)]) / dy;
        
        d_dh[IDX(i,j)] = -H * (du_dx + dv_dy);
    }
}

/*
 * CPU TO GPU KERNEL TRANSLATION - Velocity Derivatives
 * Original CPU functions were similar to compute_dh() but for u and v components
 * Same parallelization strategy applied:
 * 1. One thread per grid point
 * 2. Direct array indexing instead of helper functions
 * 3. Explicit parameter passing
 */
__global__ void compute_du_kernel(double *d_h, double *d_du, 
                                int nx, int ny, double g, double dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {
        double dh_dx = (d_h[IDX(i+1,j)] - d_h[IDX(i,j)]) / dx;
        d_du[IDX(i,j)] = -g * dh_dx;
    }
}

__global__ void compute_dv_kernel(double *d_h, double *d_dv, 
                                int nx, int ny, double g, double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {
        double dh_dy = (d_h[IDX(i,j+1)] - d_h[IDX(i,j)]) / dy;
        d_dv[IDX(i,j)] = -g * dh_dy;
    }
}

/*
 * CPU TO GPU KERNEL TRANSLATION - Multistep Update
 * Original CPU code had separate loops updating h, u, and v fields
 * GPU version:
 * 1. Combines updates into single kernel for better efficiency
 * 2. Each thread updates one grid point for all fields
 * 3. Careful handling of array bounds for u and v updates
 */
__global__ void multistep_kernel(double *d_h, double *d_u, double *d_v,
                                double *d_dh, double *d_du, double *d_dv,
                                double *d_dh1, double *d_du1, double *d_dv1,
                                double *d_dh2, double *d_du2, double *d_dv2,
                                int nx, int ny, double dt,
                                double a1, double a2, double a3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {
        int idx = IDX(i,j);
        // Update all fields at this grid point
        d_h[idx] += (a1 * d_dh[idx] + a2 * d_dh1[idx] + a3 * d_dh2[idx]) * dt;
        
        // Handle boundary cases for velocity updates
        if (i < nx-1) d_u[IDX(i+1,j)] += (a1 * d_du[idx] + a2 * d_du1[idx] + a3 * d_du2[idx]) * dt;
        if (j < ny-1) d_v[IDX(i,j+1)] += (a1 * d_dv[idx] + a2 * d_dv1[idx] + a3 * d_dv2[idx]) * dt;
    }
}

/*
 * CPU TO GPU KERNEL TRANSLATION - Boundary Conditions
 * Original CPU code had 4 separate functions:
 * - compute_ghost_horizontal()
 * - compute_ghost_vertical()
 * - compute_boundaries_horizontal()
 * - compute_boundaries_vertical()
 *
 * GPU version:
 * 1. Combines all boundary handling into single kernel
 * 2. Uses 1D thread arrangement since boundaries are 1D
 * 3. Each thread handles one boundary point
 */
__global__ void compute_boundaries_kernel(double *d_h, double *d_u, double *d_v, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handle horizontal boundaries
    if (idx < ny) {
        d_h[IDX(nx,idx)] = d_h[IDX(0,idx)];
        d_u[IDX(0,idx)] = d_u[IDX(nx,idx)];
    }
    
    // Handle vertical boundaries
    if (idx < nx) {
        d_h[IDX(idx,ny)] = d_h[IDX(idx,0)];
        d_v[IDX(idx,0)] = d_v[IDX(idx,ny)];
    }
}

/*
 * CPU TO GPU TRANSLATION - Initialization
 * Changes from CPU version:
 * 1. Allocate GPU memory for all arrays
 * 2. Copy initial conditions from CPU to GPU
 * 3. Initialize derivative arrays on GPU
 */
void init(double *h0, double *u0, double *v0, double length_, double width_, 
          int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_) {
    // Store grid parameters (same as CPU version)
    nx = nx_;
    ny = ny_;
    H = H_;
    g = g_;
    dx = length_ / nx;
    dy = width_ / ny;
    dt = dt_;
    
    // Allocate device memory for all arrays
    size_t size = nx * ny * sizeof(double);
    cudaMalloc(&d_h, size);
    cudaMalloc(&d_u, size);
    cudaMalloc(&d_v, size);
    cudaMalloc(&d_dh, size);
    cudaMalloc(&d_du, size);
    cudaMalloc(&d_dv, size);
    cudaMalloc(&d_dh1, size);
    cudaMalloc(&d_du1, size);
    cudaMalloc(&d_dv1, size);
    cudaMalloc(&d_dh2, size);
    cudaMalloc(&d_du2, size);
    cudaMalloc(&d_dv2, size);
    
    // Copy initial conditions to GPU
    cudaMemcpy(d_h, h0, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u0, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v0, size, cudaMemcpyHostToDevice);
    
    // Initialize derivative arrays to zero
    cudaMemset(d_dh, 0, size);
    cudaMemset(d_du, 0, size);
    cudaMemset(d_dv, 0, size);
    cudaMemset(d_dh1, 0, size);
    cudaMemset(d_du1, 0, size);
    cudaMemset(d_dv1, 0, size);
    cudaMemset(d_dh2, 0, size);
    cudaMemset(d_du2, 0, size);
    cudaMemset(d_dv2, 0, size);
}

/*
 * CPU TO GPU TRANSLATION - Time Stepping
 * Changes from CPU version:
 * 1. Replace direct function calls with kernel launches
 * 2. Configure grid and block dimensions for kernels
 * 3. Keep buffer swapping logic similar to CPU
 */
void step() {
    // Configure thread blocks and grid for 2D kernels
    dim3 block(16, 16);  // 16x16 threads per block is typically efficient
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    
    // Launch computation kernels
    compute_dh_kernel<<<grid, block>>>(d_h, d_u, d_v, d_dh, nx, ny, H, dx, dy);
    compute_du_kernel<<<grid, block>>>(d_h, d_du, nx, ny, g, dx);
    compute_dv_kernel<<<grid, block>>>(d_h, d_dv, nx, ny, g, dy);
    
    // Set multistep coefficients (same as CPU version)
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
    multistep_kernel<<<grid, block>>>(d_h, d_u, d_v, d_dh, d_du, d_dv,
                                     d_dh1, d_du1, d_dv1, d_dh2, d_du2, d_dv2,
                                     nx, ny, dt, a1, a2, a3);
    
    // Handle boundaries with 1D thread configuration
    dim3 boundary_block(256);  // Use 256 threads per block for 1D operations
    dim3 boundary_grid((max(nx, ny) + boundary_block.x - 1) / boundary_block.x);
    compute_boundaries_kernel<<<boundary_grid, boundary_block>>>(d_h, d_u, d_v, nx, ny);
    
    // Swap derivative buffers (similar to CPU version)
    double *tmp;
    tmp = d_dh2; d_dh2 = d_dh1; d_dh1 = d_dh; d_dh = tmp;
    tmp = d_du2; d_du2 = d_du1; d_du1 = d_du; d_du = tmp;
    tmp = d_dv2; d_dv2 = d_dv1; d_dv1 = d_dv; d_dv = tmp;
    
    t++;
}

/*
 * CPU TO GPU TRANSLATION - Data Transfer
 * New function needed for GPU version to copy results back to CPU
 */
void transfer(double *h_host) {
    cudaMemcpy(h_host, d_h, nx * ny * sizeof(double), cudaMemcpyDeviceToHost);
}

/*
 * CPU TO GPU TRANSLATION - Cleanup
 * New function needed for GPU version to free device memory
 */
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