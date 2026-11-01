// Day 15-16: standalone CUDA, independent of Python/PyTorch and their C++ ABI.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); \
        std::exit(1); \
    } \
} while (0)

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void block_sum(const float* a, float* partial, int n) {
    __shared__ float values[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    values[threadIdx.x] = i < n ? a[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = values[0];
}

int main() {
    for (int n : {1, 255, 256, 257, 1003, 65537}) {
        int blocks = (n + 255) / 256;
        std::vector<float> a(n), b(n), out(n), partial(blocks);
        for (int i = 0; i < n; ++i) {
            a[i] = float(i % 17 - 8) / 8;
            b[i] = float(i % 13 - 6) / 8;
        }
        float *da, *db, *dc, *dp;
        CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dp, blocks * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        vector_add<<<blocks, 256>>>(da, db, dc, n);
        CUDA_CHECK(cudaGetLastError());
        block_sum<<<blocks, 256>>>(da, dp, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(partial.data(), dp, blocks * sizeof(float), cudaMemcpyDeviceToHost));
        double expected = 0, actual = 0;
        for (int i = 0; i < n; ++i) {
            expected += a[i];
            if (std::abs(out[i] - a[i] - b[i]) > 1e-6) return 2;
        }
        for (float value : partial) actual += value;
        if (std::abs(actual - expected) > 1e-5) return 3;
        CUDA_CHECK(cudaFree(da));
        CUDA_CHECK(cudaFree(db));
        CUDA_CHECK(cudaFree(dc));
        CUDA_CHECK(cudaFree(dp));
    }
    std::puts("{\"vector_add\":\"passed\",\"shared_memory_reduction\":\"passed\",\"sizes\":6}");
}
