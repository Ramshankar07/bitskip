#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for in-place FWHT along last dimension. Assumes power-of-two length.
template <typename scalar_t>
__global__ void fwht_kernel(scalar_t* data, int rows, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  scalar_t* x = data + row * n;
  for (int h = 1; h < n; h <<= 1) {
    for (int i = 0; i < n; i += h << 1) {
      for (int j = i; j < i + h; ++j) {
        scalar_t a = x[j];
        scalar_t b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
      }
    }
  }
}

torch::Tensor fwht_cuda(torch::Tensor x) {
  auto x_contig = x.contiguous();
  int64_t n = x_contig.size(-1);
  TORCH_CHECK((n & (n - 1)) == 0, "FWHT requires power-of-two length");

  auto y = x_contig.view({-1, n}).contiguous();
  int rows = y.size(0);

  int threads = 128;
  int blocks = (rows + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "fwht_cuda", ([&] {
    fwht_kernel<scalar_t><<<blocks, threads>>>(y.data_ptr<scalar_t>(), rows, (int)n);
  }));

  return y.view(x_contig.sizes());
}


