#include "attention_scores.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <float.h>

// Kernel for computing Q @ K^T with scaling
__global__ void attention_scores_kernel(
    const float* __restrict__ q,      // [batch_size, num_heads, seq_len_q, head_dim]
    const float* __restrict__ k,      // [batch_size, num_heads, seq_len_k, head_dim]
    const float* __restrict__ mask,   // [batch_size, 1, 1, seq_len_k] or nullptr
    float* __restrict__ scores,       // [batch_size, num_heads, seq_len_q, seq_len_k]
    const float scale,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int head_dim
) {
    // Tile dimensions optimized for attention computation
    constexpr int TILE_SIZE = 16;
    
    // Shared memory for tiles
    __shared__ float q_tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    __shared__ float k_tile[TILE_SIZE][TILE_SIZE + 1];
    
    // Thread and block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Calculate which attention score this thread computes
    const int batch_idx = blockIdx.z / num_heads;
    const int head_idx = blockIdx.z % num_heads;
    const int row = blockIdx.y * TILE_SIZE + ty;  // seq_len_q dimension
    const int col = blockIdx.x * TILE_SIZE + tx;  // seq_len_k dimension
    
    // Accumulator for dot product
    float sum = 0.0f;
    
    // Base pointers for this batch and head
    const int q_base = (batch_idx * num_heads + head_idx) * seq_len_q * head_dim;
    const int k_base = (batch_idx * num_heads + head_idx) * seq_len_k * head_dim;
    
    // Compute dot product Q[row] · K[col] using tiling
    for (int tile = 0; tile < (head_dim + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load Q tile
        int q_row = row;
        int q_col = tile * TILE_SIZE + tx;
        if (q_row < seq_len_q && q_col < head_dim) {
            q_tile[ty][tx] = q[q_base + q_row * head_dim + q_col];
        } else {
            q_tile[ty][tx] = 0.0f;
        }
        
        // Load K tile (note: we're accessing K in transposed manner)
        int k_row = col;
        int k_col = tile * TILE_SIZE + ty;
        if (k_row < seq_len_k && k_col < head_dim) {
            k_tile[ty][tx] = k[k_base + k_row * head_dim + k_col];
        } else {
            k_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += q_tile[ty][i] * k_tile[i][tx];
        }
        
        __syncthreads();
    }
    
    // Write result with scaling and optional masking
    if (row < seq_len_q && col < seq_len_k) {
        // Apply scale
        float score = sum * scale;
        
        // Apply attention mask if provided
        if (mask != nullptr) {
            // Mask shape: [batch_size, 1, 1, seq_len_k]
            float mask_val = mask[batch_idx * seq_len_k + col];
            // If mask is 0, add large negative value
            score += (1.0f - mask_val) * -10000.0f;
        }
        
        // Write to output
        const int out_idx = ((batch_idx * num_heads + head_idx) * seq_len_q + row) * seq_len_k + col;
        scores[out_idx] = score;
    }
}

// Alternative kernel for small sequence lengths using different memory access pattern
__global__ void attention_scores_kernel_small_seq(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ mask,
    float* __restrict__ scores,
    const float scale,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int head_dim
) {
    // Each thread handles one attention score
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_scores = batch_size * num_heads * seq_len_q * seq_len_k;
    
    if (tid >= total_scores) return;
    
    // Decompose thread ID into indices
    const int seq_k_idx = tid % seq_len_k;
    const int seq_q_idx = (tid / seq_len_k) % seq_len_q;
    const int head_idx = (tid / (seq_len_k * seq_len_q)) % num_heads;
    const int batch_idx = tid / (seq_len_k * seq_len_q * num_heads);
    
    // Compute dot product
    float sum = 0.0f;
    const int q_offset = ((batch_idx * num_heads + head_idx) * seq_len_q + seq_q_idx) * head_dim;
    const int k_offset = ((batch_idx * num_heads + head_idx) * seq_len_k + seq_k_idx) * head_dim;
    
    #pragma unroll 8
    for (int i = 0; i < head_dim; i++) {
        sum += q[q_offset + i] * k[k_offset + i];
    }
    
    // Apply scale
    sum *= scale;
    
    // Apply mask if provided
    if (mask != nullptr) {
        float mask_val = mask[batch_idx * seq_len_k + seq_k_idx];
        sum += (1.0f - mask_val) * -10000.0f;
    }
    
    scores[tid] = sum;
}

extern "C" void launch_attention_scores(
    const float* q,
    const float* k,
    const float* mask,
    float* scores,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
) {
    // Choose kernel based on sequence length
    if (seq_len_q <= 128 && seq_len_k <= 128) {
        // Use simple kernel for small sequences
        int total_scores = batch_size * num_heads * seq_len_q * seq_len_k;
        int threads = 256;
        int blocks = (total_scores + threads - 1) / threads;
        
        attention_scores_kernel_small_seq<<<blocks, threads>>>(
            q, k, mask, scores, scale,
            batch_size, num_heads, seq_len_q, seq_len_k, head_dim
        );
    } else {
        // Use tiled kernel for larger sequences
        constexpr int TILE_SIZE = 16;
        dim3 grid(
            (seq_len_k + TILE_SIZE - 1) / TILE_SIZE,
            (seq_len_q + TILE_SIZE - 1) / TILE_SIZE,
            batch_size * num_heads
        );
        dim3 block(TILE_SIZE, TILE_SIZE);
        
        attention_scores_kernel<<<grid, block>>>(
            q, k, mask, scores, scale,
            batch_size, num_heads, seq_len_q, seq_len_k, head_dim
        );
    }
    
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

// Python/PyTorch binding helper
extern "C" void attention_scores_cuda(
    const float* q,      // [batch_size, num_heads, seq_len_q, head_dim]
    const float* k,      // [batch_size, num_heads, seq_len_k, head_dim]
    const float* mask,   // [batch_size, 1, 1, seq_len_k] or nullptr
    float* scores,       // [batch_size, num_heads, seq_len_q, seq_len_k]
    float scale,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
) {
    launch_attention_scores(q, k, mask, scores, scale, 
                          batch_size, num_heads, seq_len_q, seq_len_k, head_dim);
}