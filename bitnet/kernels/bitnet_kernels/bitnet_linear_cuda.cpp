#include <torch/extension.h>
#include "bitnet_linear.h"

torch::Tensor bitlinear_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias
) {
    // Get dimensions
    const int M = x.size(0);
    const int N = weight.size(0);
    const int K = weight.size(1);
    
    // Create output tensor
    auto output = torch::empty({M, N}, x.options());
    
    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();
    
    // Launch CUDA kernel
    launch_linear(x_ptr, weight_ptr, bias_ptr, output_ptr, M, N, K);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitlinear_forward_cuda", &bitlinear_forward_cuda, "BitLinear forward (CUDA)");
} 