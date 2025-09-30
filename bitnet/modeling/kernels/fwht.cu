// ============================================================================
// fwht.cu - Simple, correct CUDA kernel for Fast Walsh-Hadamard Transform
// ============================================================================

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Simple, correct FWHT kernel
template <typename scalar_t>
__global__ void fwht_kernel(
    scalar_t* __restrict__ data,
    const int rows,
    const int n
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    scalar_t* x = data + row * n;
    
    // Use shared memory
    extern __shared__ char shared_mem[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(shared_mem);
    
    // Load data into shared memory
    for (int i = tid; i < n; i += blockDim.x) {
        shared[i] = x[i];
    }
    __syncthreads();
    
    // Perform FWHT - only thread 0 does the computation
    if (tid == 0) {
        for (int len = 1; len < n; len <<= 1) {
            for (int i = 0; i < n; i += (len << 1)) {
                for (int j = i; j < i + len; ++j) {
                    scalar_t a = shared[j];
                    scalar_t b = shared[j + len];
                    shared[j] = a + b;
                    shared[j + len] = a - b;
                }
            }
        }
    }
    __syncthreads();
    
    // Write back to global memory
    for (int i = tid; i < n; i += blockDim.x) {
        x[i] = shared[i];
    }
}

// Main CUDA function
torch::Tensor fwht_cuda_forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    const int64_t n = x.size(-1);
    
    // Validate input
    TORCH_CHECK((n & (n - 1)) == 0, "FWHT requires power-of-two length, got ", n);
    TORCH_CHECK(n >= 2 && n <= 65536, "FWHT supports sizes 2 to 65536, got ", n);
    
    // Clone input to avoid in-place modification issues
    auto y = x.clone();
    
    // Flatten all dimensions except the last
    auto y_flat = y.view({-1, n}).contiguous();
    const int rows = (int) y_flat.size(0);
    
    // Launch kernel
    const int threads = min(512, max(32, (int)n));
    const int blocks = rows;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        y_flat.scalar_type(), "fwht_cuda", ([&] {
            size_t shared_bytes = (size_t)n * sizeof(scalar_t);
            fwht_kernel<scalar_t><<<blocks, threads, shared_bytes>>>(
                y_flat.data_ptr<scalar_t>(),
                rows,
                (int)n
            );
        })
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA error: ", cudaGetErrorString(error));
    
    return y.view(x.sizes());
}

// Backward pass (FWHT is self-inverse up to scaling)
torch::Tensor fwht_cuda_backward(torch::Tensor grad_output) {
    return fwht_cuda_forward(grad_output);
}


