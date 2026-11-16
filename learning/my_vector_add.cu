// P2-01 personal kernel. Keep the function name and signature unchanged.
// The host harness owns allocation, copies, launch dimensions and validation.
#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    // V1: compute the global element index i.
    // blockIdx.x: block number; blockDim.x: threads per block;
    // threadIdx.x: this thread's index within its block.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // V2: only when i < n, write out[i] = a[i] + b[i].
    // Do not read or write the tail, and do not change a or b.
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
