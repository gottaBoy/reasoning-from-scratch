// P2-01 reference kernel; the independent host harness supplies main().
#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
