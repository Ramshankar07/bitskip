#include <torch/extension.h>
#include "attention_scores.h"

torch::Tensor attention_scores_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& mask,
    float scale
) {
    // Get tensor dimensions
    const int batch_size = q.size(0);
    const int num_heads = q.size(1);
    const int seq_len_q = q.size(2);
    const int seq_len_k = k.size(2);
    const int head_dim = q.size(3);
    
    // Verify tensor shapes
    TORCH_CHECK(q.dim() == 4, "Query tensor must be 4D: [batch_size, num_heads, seq_len_q, head_dim]");
    TORCH_CHECK(k.dim() == 4, "Key tensor must be 4D: [batch_size, num_heads, seq_len_k, head_dim]");
    TORCH_CHECK(q.size(0) == k.size(0), "Batch sizes must match");
    TORCH_CHECK(q.size(1) == k.size(1), "Number of heads must match");
    TORCH_CHECK(q.size(3) == k.size(3), "Head dimensions must match");
    
    // Create output tensor
    auto scores = torch::empty({batch_size, num_heads, seq_len_q, seq_len_k}, q.options());
    
    // Get raw pointers
    const float* q_ptr = q.data_ptr<float>();
    const float* k_ptr = k.data_ptr<float>();
    const float* mask_ptr = mask.defined() ? mask.data_ptr<float>() : nullptr;
    float* scores_ptr = scores.data_ptr<float>();
    
    // Launch CUDA kernel
    attention_scores_cuda(q_ptr, k_ptr, mask_ptr, scores_ptr, scale,
                         batch_size, num_heads, seq_len_q, seq_len_k, head_dim);
    
    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_scores_forward", &attention_scores_forward, "Attention scores computation (CUDA)");
} 