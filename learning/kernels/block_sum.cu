// P2-02 reference: exactly 256 threads per block.
#include <cuda_runtime.h>

__global__ void block_sum(const float* input, float* partial, int n) {
    __shared__ float values[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    values[tid] = i < n ? input[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) values[tid] += values[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = values[0];
}
