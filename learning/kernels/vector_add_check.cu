// P2-01 host-side oracle. Link with a trusted reference or learner kernel.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* out, int n);

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        std::exit(1); \
    } \
} while (0)

struct Result {
    int n;
    int blocks;
    int value_mismatches = 0;
    int guard_mismatches = 0;
    int input_mismatches = 0;
    bool passed() const {
        return value_mismatches == 0 && guard_mismatches == 0 && input_mismatches == 0;
    }
};

Result check_size(int n) {
    constexpr int threads = 256;
    constexpr int guard = 8;
    constexpr float sentinel = -12345.5f;
    Result result{n, (n + threads - 1) / threads};
    const int allocated = result.blocks * threads + 2 * guard;
    const size_t bytes = allocated * sizeof(float);
    std::vector<float> a(allocated), b(allocated), output(allocated, sentinel);
    std::vector<float> after_a(allocated), after_b(allocated);
    for (int i = 0; i < allocated; ++i) {
        a[i] = float(i % 17 - 8) / 8;
        b[i] = float(i % 13 - 6) / 8;
    }
    // A no-op kernel must not accidentally pass by reading uninitialized output.
    std::fill(output.begin() + guard, output.begin() + guard + n,
              std::numeric_limits<float>::quiet_NaN());
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, output.data(), bytes, cudaMemcpyHostToDevice));
    vector_add<<<result.blocks, threads>>>(da + guard, db + guard, dc + guard, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), dc, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(after_a.data(), da, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(after_b.data(), db, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < allocated; ++i) {
        if (i >= guard && i < guard + n) {
            double expected = double(a[i]) + double(b[i]);
            if (!std::isfinite(output[i]) || std::abs(double(output[i]) - expected) > 1e-6)
                ++result.value_mismatches;
        } else if (output[i] != sentinel) {
            ++result.guard_mismatches;
        }
        if (after_a[i] != a[i] || after_b[i] != b[i]) ++result.input_mismatches;
    }
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    return result;
}

int main() {
    int device;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::vector<Result> results;
    int passed = 0;
    for (int n : {1, 255, 256, 257, 1003, 65537}) {
        results.push_back(check_size(n));
        passed += results.back().passed();
    }
    std::printf("{\"practice\":\"P2-01\",\"status\":\"%s\",\"passed\":%d,"
                "\"failed\":%d,\"capability\":[%d,%d],\"threads_per_block\":256,\"cases\":[",
                passed == 6 ? "passed" : "failed", passed, 6 - passed,
                properties.major, properties.minor);
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        std::printf("%s{\"n\":%d,\"blocks\":%d,\"status\":\"%s\","
                    "\"value_mismatches\":%d,\"guard_mismatches\":%d,\"input_mismatches\":%d}",
                    i ? "," : "", r.n, r.blocks, r.passed() ? "passed" : "failed",
                    r.value_mismatches, r.guard_mismatches, r.input_mismatches);
    }
    std::puts("]}");
    return passed == 6 ? 0 : 2;
}
