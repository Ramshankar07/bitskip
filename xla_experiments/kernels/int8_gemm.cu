// INT8 GEMM kernel with dequantization
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

template <typename scalar_t>
__global__ void int8_gemm_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const scalar_t* __restrict__ scale_a,
    const scalar_t* __restrict__ scale_b,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int32_t sum = 0;
        for (int k = 0; k < K; k++) {
            sum += (int32_t)A[row * K + k] * (int32_t)B[k * N + col];
        }
        C[row * N + col] = (scalar_t)sum * scale_a[row] * scale_b[col];
    }
}

torch::Tensor int8_gemm_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor scale_a,
    torch::Tensor scale_b
) {
    CHECK_CUDA(a);
    CHECK_CUDA(b);
    CHECK_CONTIGUOUS(a);
    CHECK_CONTIGUOUS(b);
    
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
    auto c = torch::zeros({M, N}, options);
    
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(c.scalar_type(), "int8_gemm", ([&] {
        int8_gemm_kernel<scalar_t><<<grid, block>>>(
            a.data_ptr<int8_t>(),
            b.data_ptr<int8_t>(),
            c.data_ptr<scalar_t>(),
            scale_a.data_ptr<scalar_t>(),
            scale_b.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return c;
}

