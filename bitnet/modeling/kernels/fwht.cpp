#include <torch/extension.h>

// Forward declaration of CUDA implementation
torch::Tensor fwht_cuda(torch::Tensor x);

torch::Tensor fwht(torch::Tensor x) {
  TORCH_CHECK(x.device().is_cuda(), "fwht: expected CUDA tensor");
  return fwht_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwht", &fwht, "Fast Walsh-Hadamard Transform (CUDA)");
}


