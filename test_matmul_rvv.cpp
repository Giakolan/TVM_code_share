#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

#include <dlpack/dlpack.h>

extern "C" void matmul(
    std::vector<const DLTensor*>& data_entry_,
    std::vector<int64_t>& shapeA,
    std::vector<int64_t>& shapeB
);

static DLTensor MakeTensor(
    float* data,
    int ndim,
    int64_t* shape
) {
    DLTensor tensor{};

    tensor.data = data;
    tensor.device.device_type = kDLCPU;
    tensor.device.device_id = 0;
    tensor.ndim = ndim;

    tensor.dtype.code = kDLFloat;
    tensor.dtype.bits = 32;
    tensor.dtype.lanes = 1;

    tensor.shape = shape;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;

    return tensor;
}

static void ScalarMatmul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M,
    int K,
    int N
) {
    std::fill(C.begin(), C.end(), 0.0f);

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const float a = A[static_cast<size_t>(i) * K + k];

            for (int j = 0; j < N; ++j) {
                C[static_cast<size_t>(i) * N + j] +=
                    a * B[static_cast<size_t>(k) * N + j];
            }
        }
    }
}

static bool RunOneTest(
    int M,
    int K,
    int N,
    std::mt19937& generator
) {
    const size_t a_size = static_cast<size_t>(M) * K;
    const size_t b_size = static_cast<size_t>(K) * N;
    const size_t c_size = static_cast<size_t>(M) * N;

    std::vector<float> A(a_size);
    std::vector<float> B(b_size);
    std::vector<float> C_rvv(c_size, 0.0f);
    std::vector<float> C_ref(c_size, 0.0f);

    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (float& value : A) {
        value = distribution(generator);
    }

    for (float& value : B) {
        value = distribution(generator);
    }

    ScalarMatmul(A, B, C_ref, M, K, N);

    int64_t A_shape[2] = {M, K};
    int64_t B_shape[2] = {K, N};
    int64_t C_shape[2] = {M, N};

    DLTensor A_tensor = MakeTensor(A.data(), 2, A_shape);
    DLTensor B_tensor = MakeTensor(B.data(), 2, B_shape);
    DLTensor C_tensor = MakeTensor(C_rvv.data(), 2, C_shape);

    std::vector<const DLTensor*> entries = {
        &A_tensor,
        &B_tensor,
        &C_tensor
    };

    std::vector<int64_t> shapeA = {M, K};
    std::vector<int64_t> shapeB = {K, N};

    matmul(entries, shapeA, shapeB);

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    size_t error_index = 0;

    for (size_t index = 0; index < c_size; ++index) {
        const float abs_error =
            std::fabs(C_rvv[index] - C_ref[index]);

        const float denominator =
            std::max(std::fabs(C_ref[index]), 1e-6f);

        const float rel_error = abs_error / denominator;

        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            error_index = index;
        }

        max_rel_error = std::max(max_rel_error, rel_error);
    }

    const float abs_tolerance = 1e-3f;
    const float rel_tolerance = 1e-3f;

    const bool passed =
        max_abs_error <= abs_tolerance ||
        max_rel_error <= rel_tolerance;

    std::cout
        << "M=" << M
        << " K=" << K
        << " N=" << N
        << " | max_abs_error=" << max_abs_error
        << " | max_rel_error=" << max_rel_error
        << " | " << (passed ? "PASS" : "FAIL")
        << '\n';

    if (!passed) {
        const int row = static_cast<int>(error_index / N);
        const int col = static_cast<int>(error_index % N);

        std::cerr
            << "  Largest error at C[" << row << "][" << col << "]\n"
            << "  RVV       = " << C_rvv[error_index] << '\n'
            << "  Reference = " << C_ref[error_index] << '\n';
    }

    return passed;
}

static void RunBenchmark(
    int M,
    int K,
    int N,
    int warmup_runs,
    int measured_runs,
    std::mt19937& generator
) {
    const size_t a_size = static_cast<size_t>(M) * K;
    const size_t b_size = static_cast<size_t>(K) * N;
    const size_t c_size = static_cast<size_t>(M) * N;

    std::vector<float> A(a_size);
    std::vector<float> B(b_size);
    std::vector<float> C(c_size, 0.0f);

    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (float& value : A) {
        value = distribution(generator);
    }

    for (float& value : B) {
        value = distribution(generator);
    }

    int64_t A_shape[2] = {M, K};
    int64_t B_shape[2] = {K, N};
    int64_t C_shape[2] = {M, N};

    DLTensor A_tensor = MakeTensor(A.data(), 2, A_shape);
    DLTensor B_tensor = MakeTensor(B.data(), 2, B_shape);
    DLTensor C_tensor = MakeTensor(C.data(), 2, C_shape);

    std::vector<const DLTensor*> entries = {
        &A_tensor,
        &B_tensor,
        &C_tensor
    };

    std::vector<int64_t> shapeA = {M, K};
    std::vector<int64_t> shapeB = {K, N};

    // Warm-up：避免第一次執行的初始化成本干擾
    for (int run = 0; run < warmup_runs; ++run) {
        matmul(entries, shapeA, shapeB);
    }

    std::vector<double> elapsed_times;
    elapsed_times.reserve(measured_runs);

    for (int run = 0; run < measured_runs; ++run) {
        const auto start = std::chrono::steady_clock::now();

        matmul(entries, shapeA, shapeB);

        const auto end = std::chrono::steady_clock::now();

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        elapsed_times.push_back(elapsed_ms);
    }

    std::sort(elapsed_times.begin(), elapsed_times.end());

    double total_ms = 0.0;

    for (double time : elapsed_times) {
        total_ms += time;
    }

    const double average_ms =
        total_ms / static_cast<double>(measured_runs);

    const double median_ms =
        elapsed_times[measured_runs / 2];

    const double min_ms = elapsed_times.front();

    // Matrix multiplication FLOPs 約為 2*M*K*N
    const double operations =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(K) *
        static_cast<double>(N);

    const double gflops =
        operations / (median_ms * 1.0e6);

    std::cout
        << "M=" << std::setw(4) << M
        << " K=" << std::setw(4) << K
        << " N=" << std::setw(4) << N
        << " | avg=" << std::fixed << std::setprecision(3)
        << average_ms << " ms"
        << " | median=" << median_ms << " ms"
        << " | min=" << min_ms << " ms"
        << " | " << gflops << " GFLOPS"
        << '\n';
}


int main() {
    std::mt19937 generator(12345);

    struct Shape {
        int M;
        int K;
        int N;
    };

    const std::vector<Shape> tests = {
        {2, 3, 2},
        {1, 384, 384},
        {1, 384, 1536},
        {1, 1536, 384},
        {10, 384, 384},
        {64, 384, 384},

        // 測試不足 tile size 的尾端
        {3, 7, 5},
        {65, 129, 130}
    };

    bool all_passed = true;

    for (const Shape& shape : tests) {
        const bool passed =
            RunOneTest(shape.M, shape.K, shape.N, generator);

        all_passed = all_passed && passed;
    }

    if (!all_passed) {
        std::cerr << "[FAIL] Some RVV matmul tests failed.\n";
        return 1;
    }

    std::cout << "\n===== RVV MATMUL M=1 SPECIALIZED BENCHMARK =====\n";

    const int warmup_runs = 5;
    const int measured_runs = 20;

    RunBenchmark(
        1, 384, 384,
        warmup_runs,
        measured_runs,
        generator
    );

    RunBenchmark(
        1, 384, 1536,
        warmup_runs,
        measured_runs,
        generator
    );

    RunBenchmark(
        1, 1536, 384,
        warmup_runs,
        measured_runs,
        generator
    );

    RunBenchmark(
        10, 384, 384,
        warmup_runs,
        measured_runs,
        generator
    );

    RunBenchmark(
        64, 384, 384,
        warmup_runs,
        measured_runs,
        generator
    );

    std::cout << "[PASS] All RVV matmul tests passed.\n";
    return 0;
}