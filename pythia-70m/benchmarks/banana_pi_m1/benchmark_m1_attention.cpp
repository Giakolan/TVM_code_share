#include <dlpack/dlpack.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


using MatmulFn =
    void (*)(
        std::vector<const DLTensor*>&,
        std::vector<int64_t>&,
        std::vector<int64_t>&
    );


static void reference_m1(
    const float* A,
    const float* B,
    float* C,
    int batch_heads,
    int m,
    int o
) {
    for (int bh = 0; bh < batch_heads; ++bh) {

        const float* Ab =
            A + bh * m;

        const float* Bb =
            B + bh * m * o;

        float* Cb =
            C + bh * o;

        for (int j = 0; j < o; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < m; ++k) {
                sum +=
                    Ab[k] *
                    Bb[k * o + j];
            }

            Cb[j] = sum;
        }
    }
}


static void run_case(
    MatmulFn matmul,
    const std::string& label,
    const std::string& mode,
    int T,
    int repeat
) {
    constexpr int batch = 1;
    constexpr int heads = 8;
    constexpr int batch_heads = batch * heads;

    int m;
    int o;

    if (mode == "QK") {
        // M=1, K=64, N=T
        m = 64;
        o = T;
    } else {
        // M=1, K=T, N=64
        m = T;
        o = 64;
    }

    std::vector<float> A(
        batch_heads * m
    );

    std::vector<float> B(
        batch_heads * m * o
    );

    std::vector<float> C(
        batch_heads * o,
        0.0f
    );

    std::vector<float> C_ref(
        batch_heads * o,
        0.0f
    );

    for (size_t i = 0; i < A.size(); ++i) {
        A[i] =
            static_cast<float>(
                static_cast<int>(i % 17) - 8
            ) * 0.03125f;
    }

    for (size_t i = 0; i < B.size(); ++i) {
        B[i] =
            static_cast<float>(
                static_cast<int>(i % 23) - 11
            ) * 0.015625f;
    }

    int64_t shapeA[4] = {
        batch,
        heads,
        1,
        m,
    };

    int64_t shapeB[4] = {
        batch,
        heads,
        m,
        o,
    };

    int64_t shapeC[4] = {
        batch,
        heads,
        1,
        o,
    };

    DLTensor tensorA{};
    tensorA.data = A.data();
    tensorA.device = DLDevice{kDLCPU, 0};
    tensorA.ndim = 4;
    tensorA.dtype = DLDataType{kDLFloat, 32, 1};
    tensorA.shape = shapeA;
    tensorA.strides = nullptr;
    tensorA.byte_offset = 0;

    DLTensor tensorB{};
    tensorB.data = B.data();
    tensorB.device = DLDevice{kDLCPU, 0};
    tensorB.ndim = 4;
    tensorB.dtype = DLDataType{kDLFloat, 32, 1};
    tensorB.shape = shapeB;
    tensorB.strides = nullptr;
    tensorB.byte_offset = 0;

    DLTensor tensorC{};
    tensorC.data = C.data();
    tensorC.device = DLDevice{kDLCPU, 0};
    tensorC.ndim = 4;
    tensorC.dtype = DLDataType{kDLFloat, 32, 1};
    tensorC.shape = shapeC;
    tensorC.strides = nullptr;
    tensorC.byte_offset = 0;

    std::vector<const DLTensor*> entries = {
        &tensorA,
        &tensorB,
        &tensorC,
    };

    std::vector<int64_t> shape_a = {
        batch,
        heads,
        1,
        m,
    };

    std::vector<int64_t> shape_b = {
        batch,
        heads,
        m,
        o,
    };

    // -------------------------
    // Correctness
    // -------------------------

    reference_m1(
        A.data(),
        B.data(),
        C_ref.data(),
        batch_heads,
        m,
        o
    );

    matmul(
        entries,
        shape_a,
        shape_b
    );

    float max_diff = 0.0f;

    for (size_t i = 0; i < C.size(); ++i) {
        max_diff = std::max(
            max_diff,
            std::fabs(
                C[i] - C_ref[i]
            )
        );
    }

    if (max_diff > 1e-4f) {
        std::cerr
            << "FAIL correctness: "
            << label
            << " "
            << mode
            << " T="
            << T
            << " max_diff="
            << max_diff
            << std::endl;

        std::exit(1);
    }

    // -------------------------
    // Warmup
    // -------------------------

    for (int i = 0; i < 20; ++i) {
        matmul(
            entries,
            shape_a,
            shape_b
        );
    }

    // -------------------------
    // Benchmark
    // -------------------------

    auto begin =
        std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; ++i) {
        matmul(
            entries,
            shape_a,
            shape_b
        );
    }

    auto end =
        std::chrono::steady_clock::now();

    double total_us =
        std::chrono::duration<double, std::micro>(
            end - begin
        ).count();

    double average_us =
        total_us /
        static_cast<double>(repeat);

    std::cout
        << label << ","
        << mode << ","
        << T << ","
        << m << ","
        << o << ","
        << repeat << ","
        << std::fixed
        << std::setprecision(6)
        << average_us << ","
        << max_diff
        << "\n";
}


int main(int argc, char** argv) {

    if (argc < 3) {
        std::cerr
            << "Usage: "
            << argv[0]
            << " <libmatmul.so> <label> [repeat]"
            << std::endl;

        return 1;
    }

    const char* so_path = argv[1];
    std::string label = argv[2];

    int repeat = 1000;

    if (argc >= 4) {
        repeat = std::stoi(argv[3]);
    }

    void* handle = dlopen(
        so_path,
        RTLD_NOW | RTLD_LOCAL
    );

    if (!handle) {
        std::cerr
            << "dlopen failed: "
            << dlerror()
            << std::endl;

        return 1;
    }

    auto matmul =
        reinterpret_cast<MatmulFn>(
            dlsym(
                handle,
                "matmul"
            )
        );

    if (!matmul) {
        std::cerr
            << "dlsym(matmul) failed: "
            << dlerror()
            << std::endl;

        return 1;
    }

    std::cout
        << "kernel,mode,T,K,N,repeat,avg_us,max_diff\n";

    const int T_values[] = {
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
    };

    for (int T : T_values) {
        run_case(
            matmul,
            label,
            "QK",
            T,
            repeat
        );

        run_case(
            matmul,
            label,
            "PV",
            T,
            repeat
        );
    }

    dlclose(handle);

    return 0;
}
