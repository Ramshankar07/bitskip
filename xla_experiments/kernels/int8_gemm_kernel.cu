// INT8 GEMM kernel for torch.cuda._compile_kernel
// Extracted kernel function from int8_gemm.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 32;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 64;
constexpr int WARP_M = 64;
constexpr int WARP_N = 64;
constexpr int SF_BLOCK_SIZE = 16;

template <typename TypeAcc>
__device__ inline
void mma_m16n8k32_int8(const int A[4], const int B[2], TypeAcc C[4]) {
    if constexpr (std::is_same_v<TypeAcc, int>) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
              "r"(B[0]), "r"(B[1]),
              "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3])
        );
    }
}

template <typename scalar_t>
__global__ void int8_gemm_tensorcore_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const scalar_t* __restrict__ scale_a,
    const scalar_t* __restrict__ scale_b,
    int M, int N, int K
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;
    
    __shared__ int8_t smem_A[BLOCK_M * BLOCK_K];
    __shared__ int8_t smem_B[BLOCK_K * BLOCK_N];
    
    int acc[WARP_M / MMA_M][WARP_N / MMA_N][4] = {0};
    
    const int num_threads = blockDim.x;
    const int tid = threadIdx.x;
    const int A_tile_size = BLOCK_M * BLOCK_K;
    const int B_tile_size = BLOCK_K * BLOCK_N;
    
    for (int k_block = 0; k_block < K; k_block += BLOCK_K) {
        for (int i = tid; i < A_tile_size; i += num_threads) {
            int row = block_m + (i / BLOCK_K);
            int col = k_block + (i % BLOCK_K);
            smem_A[i] = (row < M && col < K) ? A[row * K + col] : 0;
        }
        
        for (int i = tid; i < B_tile_size; i += num_threads) {
            int row = k_block + (i / BLOCK_N);
            int col = block_n + (i % BLOCK_N);
            smem_B[i] = (row < K && col < N) ? B[row * N + col] : 0;
        }
        
        __syncthreads();
        
        const int warp_m = (warp_id / (BLOCK_N / WARP_N)) * WARP_M;
        const int warp_n = (warp_id % (BLOCK_N / WARP_N)) * WARP_N;
        
        for (int k_mma = 0; k_mma < BLOCK_K; k_mma += MMA_K) {
            for (int m_mma = 0; m_mma < WARP_M / MMA_M; m_mma++) {
                for (int n_mma = 0; n_mma < WARP_N / MMA_N; n_mma++) {
                    int A_frag[4], B_frag[2];
                    
                    const int a_row = warp_m + m_mma * MMA_M + (lane_id % 16);
                    const int a_col = k_mma + (lane_id / 16) * 8;
                    
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int offset = (a_row * BLOCK_K + a_col + i * 2);
                        A_frag[i] = *reinterpret_cast<const int*>(&smem_A[offset]);
                    }
                    
                    const int b_row = k_mma + (lane_id % 16) * 2;
                    const int b_col = warp_n + n_mma * MMA_N;
                    
                    #pragma unroll
                    for (int i = 0; i < 2; i++) {
                        int offset = ((b_row + i) * BLOCK_N + b_col);
                        B_frag[i] = *reinterpret_cast<const int*>(&smem_B[offset]);
                    }
                    
                    mma_m16n8k32_int8(A_frag, B_frag, acc[m_mma][n_mma]);
                }
            }
        }
        
        __syncthreads();
    }
    
    for (int m_mma = 0; m_mma < WARP_M / MMA_M; m_mma++) {
        for (int n_mma = 0; n_mma < WARP_N / MMA_N; n_mma++) {
            const int warp_m = (warp_id / (BLOCK_N / WARP_N)) * WARP_M;
            const int warp_n = (warp_id % (BLOCK_N / WARP_N)) * WARP_N;
            
            const int out_row = block_m + warp_m + m_mma * MMA_M + (lane_id / 4);
            const int out_col = block_n + warp_n + n_mma * MMA_N + (lane_id % 4) * 2;
            
            if (out_row < M && out_col < N) {
                const int m_block_idx = out_row / SF_BLOCK_SIZE;
                const int n_block_idx = out_col / SF_BLOCK_SIZE;
                const scalar_t scale = scale_a[m_block_idx] * scale_b[n_block_idx];
                C[out_row * N + out_col] = static_cast<scalar_t>(acc[m_mma][n_mma][0]) * scale;
                
                if (out_col + 1 < N) {
                    const int n_block_idx_1 = (out_col + 1) / SF_BLOCK_SIZE;
                    const scalar_t scale_1 = scale_a[m_block_idx] * scale_b[n_block_idx_1];
                    C[out_row * N + out_col + 1] = static_cast<scalar_t>(acc[m_mma][n_mma][1]) * scale_1;
                }
            }
            
            if (out_row + 8 < M && out_col < N) {
                const int m_block_idx_2 = (out_row + 8) / SF_BLOCK_SIZE;
                const int n_block_idx = out_col / SF_BLOCK_SIZE;
                const scalar_t scale_2 = scale_a[m_block_idx_2] * scale_b[n_block_idx];
                C[(out_row + 8) * N + out_col] = static_cast<scalar_t>(acc[m_mma][n_mma][2]) * scale_2;
                
                if (out_col + 1 < N) {
                    const int n_block_idx_1 = (out_col + 1) / SF_BLOCK_SIZE;
                    const scalar_t scale_3 = scale_a[m_block_idx_2] * scale_b[n_block_idx_1];
                    C[(out_row + 8) * N + out_col + 1] = static_cast<scalar_t>(acc[m_mma][n_mma][3]) * scale_3;
                }
            }
        }
    }
}

