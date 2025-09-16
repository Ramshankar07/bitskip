#ifndef BITNET_LINEAR_H
#define BITNET_LINEAR_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

// Declare the CUDA kernel function
__global__ void linear_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N, int K
);

// Declare the launch function with extern "C" for C linkage
extern "C" void launch_linear(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int M,
    int N,
    int K
);

#endif // BITNET_LINEAR_H 