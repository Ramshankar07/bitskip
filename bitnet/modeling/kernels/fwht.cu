// ============================================================================
// fwht.cu - Optimized CUDA kernel for Fast Walsh-Hadamard Transform
// ============================================================================

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized kernel using shared memory for small sizes
template <typename scalar_t>
__global__ void fwht_kernel_shared(
    scalar_t* __restrict__ data,
    const int rows,
    const int n
) {
    extern __shared__ char shared_mem[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    
    if (row >= rows) return;
    
    scalar_t* x = data + row * n;
    
    // Coalesced load into shared memory
    for (int i = tid; i < n; i += blockDim.x) {
        shared[i] = x[i];
    }
    __syncthreads();
    
    // Perform FWHT using parallel butterfly operations
    for (int len = 1; len < n; len <<= 1) {
        // Each thread handles specific butterfly pairs
        for (int i = tid; i < n/2; i += blockDim.x) {
            // Calculate indices for this butterfly
            int group = i / len;
            int idx_in_group = i % len;
            
            int idx_a = group * (2 * len) + idx_in_group;
            int idx_b = idx_a + len;
            
            scalar_t a = shared[idx_a];
            scalar_t b = shared[idx_b];
            
            shared[idx_a] = a + b;
            shared[idx_b] = a - b;
        }
        __syncthreads();
    }
    
    // Coalesced write back to global memory
    for (int i = tid; i < n; i += blockDim.x) {
        x[i] = shared[i];
    }
}

// Fallback kernel for large sizes (no shared memory)
template <typename scalar_t>
__global__ void fwht_kernel_global(
    scalar_t* __restrict__ data,
    const int rows,
    const int n
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    scalar_t* x = data + row * n;
    
    // Perform FWHT directly in global memory
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += (len << 1)) {
            for (int j = i; j < i + len; ++j) {
                scalar_t a = x[j];
                scalar_t b = x[j + len];
                x[j] = a + b;
                x[j + len] = a - b;
            }
        }
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
    
    // Determine kernel to use based on size
    const size_t shared_bytes = (size_t)n * sizeof(float);  // Approximate decision
    const bool use_shared = (shared_bytes <= 48 * 1024) && (n <= 8192);
    
    if (use_shared) {
        // Use shared memory kernel
        const int threads = min(512, max(32, (int)n / 2));
        const int blocks = rows;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            y_flat.scalar_type(), "fwht_cuda_shared", ([&] {
                size_t actual_shared_bytes = (size_t)n * sizeof(scalar_t);
                fwht_kernel_shared<scalar_t><<<blocks, threads, actual_shared_bytes>>>(
                    y_flat.data_ptr<scalar_t>(),
                    rows,
                    (int)n
                );
            })
        );
    } else {
        // Use global memory kernel
        const int threads = 256;
        const int blocks = (rows + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            y_flat.scalar_type(), "fwht_cuda_global", ([&] {
                fwht_kernel_global<scalar_t><<<blocks, threads>>>(
                    y_flat.data_ptr<scalar_t>(),
                    rows,
                    (int)n
                );
            })
        );
    }
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA error: ", cudaGetErrorString(error));
    
    return y.view(x.sizes());
}

// Backward pass (FWHT is self-inverse up to scaling)
torch::Tensor fwht_cuda_backward(torch::Tensor grad_output) {
    return fwht_cuda_forward(grad_output);
}


