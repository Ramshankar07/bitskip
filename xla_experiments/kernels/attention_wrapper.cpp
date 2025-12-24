// PyTorch bindings for attention kernels
#include <torch/extension.h>

torch::Tensor batched_qkt_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
);

torch::Tensor batched_attn_v_cuda(
    torch::Tensor attn_weights,
    torch::Tensor V,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
);

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
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_qkt", &batched_qkt_cuda, "Batched QK^T");
    m.def("batched_attn_v", &batched_attn_v_cuda, "Batched attention @ V");
    m.def("fused_attention", &fused_attention_cuda, "Fused attention kernel");
}

