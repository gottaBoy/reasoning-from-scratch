// P2-02 personal kernel: fixed launch of 256 threads per block.
// The independent host harness allocates input/partial and merges on the CPU.
#include <cuda_runtime.h>

__global__ void block_sum(const float* input, float* partial, int n) {
    // R1: declare a shared float array named values with 256 slots.
    // All threads in THIS block share it; each block has its own array.
    __shared__ float values[256];

    // R2: compute tid (local index) and i (global index).
    // Load values[tid] from input[i] when i < n; otherwise store 0.
    // Every slot must be initialized, including slots for tail threads.
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    values[tid] = i < n ? input[i] : 0.0f;

    // R3: synchronize the block after loading.
    __syncthreads();

    // R4: halve stride: 128,64,32,16,8,4,2,1.
    // For tid < stride: values[tid] += values[tid + stride].
    // Synchronize after EVERY round, outside the tid < stride branch.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) values[tid] += values[tid + stride];
        __syncthreads();
    }

    // R5: only tid == 0 writes values[0] to partial[blockIdx.x].
    // Do not merge blocks here; the host harness combines partials.
    if (tid == 0) partial[blockIdx.x] = values[0];
}
