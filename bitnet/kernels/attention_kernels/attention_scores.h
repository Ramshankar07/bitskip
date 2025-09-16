#ifndef ATTENTION_SCORES_H
#define ATTENTION_SCORES_H

#include <cuda_runtime.h>

// Declare the CUDA kernel functions
extern "C" void launch_attention_scores(
    const float* q,
    const float* k,
    const float* mask,
    float* scores,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
);

extern "C" void attention_scores_cuda(
    const float* q,
    const float* k,
    const float* mask,
    float* scores,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
);

#endif // ATTENTION_SCORES_H 