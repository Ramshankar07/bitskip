#include "bitnet_linear.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void linear_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Tile dimensions
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 16;
    
    // Shared memory buffers
    __shared__ float x_tile[TILE_M][TILE_K];
    __shared__ float w_tile[TILE_N][TILE_K];
    
    // Thread indexing
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int bid_m = blockIdx.x * TILE_M;
    const int bid_n = blockIdx.y * TILE_N;
    
    float sum = 0.0f;
    
    // Loop over K dimension in tiles
    for (int k = 0; k < K; k += TILE_K) {
        // Load x tile into shared memory
        int x_row = bid_m + tid_y;
        int x_col = k + tid_x;
        if (x_row < M && x_col < K) {
            x_tile[tid_y][tid_x] = x[x_row * K + x_col];
        } else {
            x_tile[tid_y][tid_x] = 0.0f;
        }
        
        // Load weight tile into shared memory
        int w_row = bid_n + tid_y;
        int w_col = k + tid_x;
        if (w_row < N && w_col < K) {
            w_tile[tid_y][tid_x] = weight[w_row * K + w_col];
        } else {
            w_tile[tid_y][tid_x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int i = 0; i < TILE_K; i++) {
            sum += x_tile[tid_y][i] * w_tile[tid_x][i];
        }
        
        __syncthreads();
    }
    
    // Write result to output
    int out_row = bid_m + tid_y;
    int out_col = bid_n + tid_x;
    if (out_row < M && out_col < N) {
        float val = sum;
        if (bias) {
            val += bias[out_col];
        }
        output[out_row * N + out_col] = val;
    }
}

extern "C" void launch_linear(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int M,
    int N,
    int K
) {
    // Set grid and block dimensions
    dim3 grid((M + 31) / 32, (N + 63) / 64);
    dim3 block(16, 32);  // TILE_K x TILE_M
    
    // Launch kernel
    linear_kernel<<<grid, block>>>(x, weight, bias, output, M, N, K);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    // Synchronize device
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA device synchronization error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}