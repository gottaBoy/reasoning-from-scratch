// Independent P2-02 oracle: check EVERY block partial before merging on CPU.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

__global__ void block_sum(const float* input, float* partial, int n);

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
    int partial_mismatches = 0;
    int total_mismatches = 0;
    int guard_mismatches = 0;
    int input_mismatches = 0;
    bool passed() const {
        return partial_mismatches == 0 && total_mismatches == 0
            && guard_mismatches == 0 && input_mismatches == 0;
    }
};

Result check_size(int n) {
    constexpr int threads = 256;
    constexpr int guard = 8;
    constexpr float sentinel = -12345.5f;
    Result result{n, (n + threads - 1) / threads};
    // Nonzero padding exposes reads past logical n even within an allocation.
    const int input_size = result.blocks * threads + 2 * guard;
    const int output_size = result.blocks + 2 * guard;
    const size_t input_bytes = input_size * sizeof(float);
    const size_t output_bytes = output_size * sizeof(float);
    float *di, *dp;
    CUDA_CHECK(cudaMalloc(&di, input_bytes));
    CUDA_CHECK(cudaMalloc(&dp, output_bytes));
    for (int pattern = 0; pattern < 3; ++pattern) {
        std::vector<float> input(input_size, 17.0f), after(input_size);
        std::vector<float> partial(output_size, sentinel);
        std::fill(partial.begin() + guard, partial.begin() + guard + result.blocks,
                  std::numeric_limits<float>::quiet_NaN());
        for (int i = 0; i < n; ++i) {
            input[guard + i] = pattern == 0 ? 1.0f
                : pattern == 1 ? float(i % 17 - 8) / 8
                : float((i * 37) % 101 - 50) / 13;
        }
        CUDA_CHECK(cudaMemcpy(di, input.data(), input_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dp, partial.data(), output_bytes, cudaMemcpyHostToDevice));
        block_sum<<<result.blocks, threads>>>(di + guard, dp + guard, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(partial.data(), dp, output_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after.data(), di, input_bytes, cudaMemcpyDeviceToHost));
        double expected_total = 0, actual_total = 0, total_tolerance = 0;
        for (int block = 0; block < result.blocks; ++block) {
            double expected = 0;
            for (int i = block * threads; i < std::min(n, (block + 1) * threads); ++i)
                expected += double(input[guard + i]);
            const double actual = partial[guard + block];
            const double tolerance = 1e-4 + 1e-5 * std::abs(expected);
            if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
                ++result.partial_mismatches;
            expected_total += expected;
            actual_total += actual;
            total_tolerance += tolerance;
        }
        if (!std::isfinite(actual_total)
                || std::abs(actual_total - expected_total) > total_tolerance)
            ++result.total_mismatches;
        for (int i = 0; i < output_size; ++i)
            if ((i < guard || i >= guard + result.blocks) && partial[i] != sentinel)
                ++result.guard_mismatches;
        for (int i = 0; i < input_size; ++i)
            if (after[i] != input[i]) ++result.input_mismatches;
    }
    CUDA_CHECK(cudaFree(di));
    CUDA_CHECK(cudaFree(dp));
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
    std::printf("{\"practice\":\"P2-02\",\"status\":\"%s\",\"passed\":%d,"
                "\"failed\":%d,\"capability\":[%d,%d],\"threads_per_block\":256,"
                "\"patterns\":3,\"cases\":[",
                passed == 6 ? "passed" : "failed", passed, 6 - passed,
                properties.major, properties.minor);
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        std::printf("%s{\"n\":%d,\"blocks\":%d,\"status\":\"%s\","
                    "\"partial_mismatches\":%d,\"total_mismatches\":%d,"
                    "\"guard_mismatches\":%d,\"input_mismatches\":%d}",
                    i ? "," : "", r.n, r.blocks, r.passed() ? "passed" : "failed",
                    r.partial_mismatches, r.total_mismatches,
                    r.guard_mismatches, r.input_mismatches);
    }
    std::puts("]}");
    return passed == 6 ? 0 : 2;
}
