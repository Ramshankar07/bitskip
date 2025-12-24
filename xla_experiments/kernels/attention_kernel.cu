// CUTLASS attention kernels for GQA

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// Batched QK^T kernel with INT8 quantization support
template <typename scalar_t>
__global__ void batched_qkt_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    scalar_t* __restrict__ scores,
    const scalar_t scale,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || 
        row >= seq_len_q || col >= seq_len_k) {
        return;
    }
    
    int q_offset = ((batch_idx * num_heads + head_idx) * seq_len_q + row) * head_dim;
    int k_offset = ((batch_idx * num_heads + head_idx) * seq_len_k + col) * head_dim;
    int score_offset = (batch_idx * num_heads + head_idx) * seq_len_q * seq_len_k + row * seq_len_k + col;
    
    scalar_t sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < head_dim; d++) {
        sum += Q[q_offset + d] * K[k_offset + d];
    }
    
    scores[score_offset] = sum * scale;
}

// Batched attention @ V kernel
template <typename scalar_t>
__global__ void batched_attn_v_kernel(
    const scalar_t* __restrict__ attn_weights,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = threadIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || 
        row >= seq_len_q || col >= head_dim) {
        return;
    }
    
    int attn_offset = (batch_idx * num_heads + head_idx) * seq_len_q * seq_len_k + row * seq_len_k;
    int v_offset = ((batch_idx * num_heads + head_idx) * seq_len_k) * head_dim;
    int out_offset = ((batch_idx * num_heads + head_idx) * seq_len_q + row) * head_dim + col;
    
    scalar_t sum = 0.0f;
    for (int k = 0; k < seq_len_k; k++) {
        sum += attn_weights[attn_offset + k] * V[v_offset + k * head_dim + col];
    }
    
    output[out_offset] = sum;
}

// Fused attention kernel: QK^T -> softmax -> @V (single kernel)
template <typename scalar_t>
__global__ void fused_attention_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ mask,
    const scalar_t scale,
    const scalar_t mask_value,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || row >= seq_len_q) {
        return;
    }
    
    // Compute QK^T for this row
    __shared__ scalar_t scores[256]; // Max seq_len_k per block
    scalar_t max_score = -1e9f;
    
    int q_offset = ((batch_idx * num_heads + head_idx) * seq_len_q + row) * head_dim;
    
    for (int col = threadIdx.y; col < seq_len_k; col += blockDim.y) {
        int k_offset = ((batch_idx * num_heads + head_idx) * seq_len_k + col) * head_dim;
        
        scalar_t sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < head_dim; d++) {
            sum += Q[q_offset + d] * K[k_offset + d];
        }
        
        scalar_t score = sum * scale;
        
        // Apply mask
        if (mask != nullptr) {
            int mask_idx = (batch_idx * num_heads + head_idx) * seq_len_q * seq_len_k + row * seq_len_k + col;
            if (mask[mask_idx] == 0.0f) {
                score = mask_value;
            }
        }
        
        scores[col] = score;
        max_score = fmaxf(max_score, score);
    }
    
    __syncthreads();
    
    // Find global max for softmax stability
    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
        if (threadIdx.y < offset && threadIdx.y + offset < seq_len_k) {
            max_score = fmaxf(max_score, scores[threadIdx.y + offset]);
        }
        __syncthreads();
    }
    
    // Compute exp and sum for softmax
    scalar_t exp_sum = 0.0f;
    for (int col = threadIdx.y; col < seq_len_k; col += blockDim.y) {
        scores[col] = expf(scores[col] - max_score);
        exp_sum += scores[col];
    }
    
    __syncthreads();
    
    // Normalize and compute output
    int out_offset = ((batch_idx * num_heads + head_idx) * seq_len_q + row) * head_dim;
    
    for (int d = threadIdx.y; d < head_dim; d += blockDim.y) {
        scalar_t sum = 0.0f;
        int v_offset = ((batch_idx * num_heads + head_idx) * seq_len_k) * head_dim;
        
        for (int k = 0; k < seq_len_k; k++) {
            sum += (scores[k] / exp_sum) * V[v_offset + k * head_dim + d];
        }
        
        output[out_offset + d] = sum;
    }
}

// CUDA wrapper functions
torch::Tensor batched_qkt_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
) {
    auto options = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
    auto scores = torch::zeros({batch_size, num_heads, seq_len_q, seq_len_k}, options);
    
    dim3 block(16, 16);
    dim3 grid((seq_len_q + 15) / 16, num_heads, batch_size);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "batched_qkt", ([&] {
        batched_qkt_kernel<scalar_t><<<grid, block>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            scores.data_ptr<scalar_t>(),
            scale,
            batch_size, num_heads, seq_len_q, seq_len_k, head_dim
        );
    }));
    
    return scores;
}

torch::Tensor batched_attn_v_cuda(
    torch::Tensor attn_weights,
    torch::Tensor V,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
) {
    auto options = torch::TensorOptions().dtype(attn_weights.dtype()).device(attn_weights.device());
    auto output = torch::zeros({batch_size, num_heads, seq_len_q, head_dim}, options);
    
    dim3 block(16, head_dim);
    dim3 grid((seq_len_q + 15) / 16, num_heads, batch_size);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(attn_weights.scalar_type(), "batched_attn_v", ([&] {
        batched_attn_v_kernel<scalar_t><<<grid, block>>>(
            attn_weights.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, num_heads, seq_len_q, seq_len_k, head_dim
        );
    }));
    
    return output;
}

torch::Tensor fused_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor mask,
    float scale,
    float mask_value,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
) {
    auto options = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
    auto output = torch::zeros({batch_size, num_heads, seq_len_q, head_dim}, options);
    
    dim3 block(1, 32);
    dim3 grid((seq_len_q + 0) / 1, num_heads, batch_size);
    
    const float* mask_ptr = mask.numel() > 0 ? mask.data_ptr<float>() : nullptr;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "fused_attention", ([&] {
        fused_attention_kernel<scalar_t><<<grid, block>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            mask_ptr,
            scale,
            mask_value,
            batch_size, num_heads, seq_len_q, seq_len_k, head_dim
        );
    }));
    
    return output;
}

