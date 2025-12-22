// PyTorch bindings for CUTLASS kernels
#include <torch/extension.h>

torch::Tensor int8_gemm_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor scale_a,
    torch::Tensor scale_b
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemm", &int8_gemm_cuda, "INT8 GEMM with dequantization");
}

