// ============================================================================
// fwht.cpp - C++ bindings and CPU fallback
// ============================================================================

#include <torch/extension.h>

// Declarations
torch::Tensor fwht_cuda_forward(torch::Tensor x);
torch::Tensor fwht_cuda_backward(torch::Tensor grad_output);

// CPU fallback (portable, simple)
torch::Tensor fwht_cpu(torch::Tensor x) {
  const int64_t n = x.size(-1);
  TORCH_CHECK((n & (n - 1)) == 0, "FWHT requires power-of-two length");

  auto y = x.clone();
  auto y_flat = y.view({-1, n}).contiguous();

  AT_DISPATCH_FLOATING_TYPES(y_flat.scalar_type(), "fwht_cpu", ([&] {
    auto y_accessor = y_flat.accessor<scalar_t, 2>();
    for (int64_t row = 0; row < y_flat.size(0); ++row) {
      for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += (len << 1)) {
          for (int j = i; j < i + len; ++j) {
            scalar_t a = y_accessor[row][j];
            scalar_t b = y_accessor[row][j + len];
            y_accessor[row][j] = a + b;
            y_accessor[row][j + len] = a - b;
          }
        }
      }
    }
  }));

  return y.view(x.sizes());
}

// Entry point that routes to CUDA or CPU
torch::Tensor fwht(torch::Tensor x) {
  if (x.device().is_cuda()) {
    return fwht_cuda_forward(x);
  } else {
    return fwht_cpu(x);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwht", &fwht, "Fast Walsh-Hadamard Transform");
  m.def("fwht_forward", &fwht_cuda_forward, "FWHT forward (CUDA)");
  m.def("fwht_backward", &fwht_cuda_backward, "FWHT backward (CUDA)");
}


