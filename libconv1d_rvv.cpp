#include <dlpack/dlpack.h>

#include <dlfcn.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cstdint>

using MatmulFn =
    void (*)(std::vector<const DLTensor*>&,
             std::vector<int64_t>&,
             std::vector<int64_t>&);

static void* matmul_handle = nullptr;
static MatmulFn matmul_fp = nullptr;

static void EnsureMatmulLoaded() {
    if (matmul_fp) return;

    const char* env_path =
        std::getenv("KIWIPEDIA_MATMUL_SO");

    std::vector<const char*> candidates;

    if (env_path && *env_path)
        candidates.push_back(env_path);

    candidates.push_back("/home/pi/libmatmul_conv_mt8.so");
    candidates.push_back("/home/pi/libmatmul.so");
    candidates.push_back("./libmatmul.so");

    for (const char* path : candidates) {
        matmul_handle =
            dlopen(path, RTLD_NOW | RTLD_LOCAL);

        if (!matmul_handle)
            continue;

        void* sym =
            dlsym(matmul_handle, "matmul");

        if (!sym) {
            dlclose(matmul_handle);
            matmul_handle = nullptr;
            continue;
        }

        matmul_fp =
            reinterpret_cast<MatmulFn>(sym);

        break;
    }

    if (!matmul_fp) {
        std::cerr
            << "[conv1d] failed to load matmul\n";
        std::abort();
    }
}

extern "C" void conv1d(
    std::vector<const DLTensor*>& data_entry,
    std::vector<int64_t>& input_shape,
    std::vector<int64_t>& weight_shape) {

    EnsureMatmulLoaded();

    const DLTensor* input_t  = data_entry[0];
    const DLTensor* weight_t = data_entry[1];
    const DLTensor* output_t = data_entry[2];

    const float* input =
        static_cast<const float*>(input_t->data);

    const float* weight =
        static_cast<const float*>(weight_t->data);

    float* output =
        static_cast<float*>(output_t->data);

    // only support Whisper encoder conv1d1 for now
    const int BATCH = static_cast<int>(input_shape[0]);
    const int CIN   = static_cast<int>(input_shape[1]);
    const int L     = static_cast<int>(input_shape[2]);

    const int COUT  = static_cast<int>(weight_shape[0]);
    const int KW    = static_cast<int>(weight_shape[2]);

    const int PAD = 1;

    int STRIDE;

    if (CIN == 80) {
        STRIDE = 1;
    } else if (CIN == 384) {
        STRIDE = 2;
    } else {
        std::cerr << "[conv1d] unsupported CIN=" << CIN << "\n";
        std::abort();
    }

    const int OUT =
        (L + 2 * PAD - KW) / STRIDE + 1;

    bool is_conv1 =
        (BATCH == 1 &&
        CIN == 80 &&
        COUT == 384 &&
        KW == 3 &&
        STRIDE == 1 &&
        OUT == L);

    bool is_conv2 =
        (BATCH == 1 &&
        CIN == 384 &&
        COUT == 384 &&
        KW == 3 &&
        STRIDE == 2 &&
        OUT == (L + 1) / 2);

    if (!is_conv1 && !is_conv2) {
        std::cerr
            << "[conv1d] unsupported shape: "
            << "B=" << BATCH
            << " CIN=" << CIN
            << " L=" << L
            << " COUT=" << COUT
            << " KW=" << KW
            << " STRIDE=" << STRIDE
            << " OUT=" << OUT
            << "\n";

        std::abort();
    }

    if (output_t->ndim != 3 ||
        output_t->shape[0] != BATCH ||
        output_t->shape[1] != COUT ||
        output_t->shape[2] != OUT) {

        std::cerr
            << "[conv1d] output shape mismatch: expected=("
            << BATCH << ", "
            << COUT << ", "
            << OUT << ") actual=("
            << output_t->shape[0] << ", "
            << output_t->shape[1] << ", "
            << output_t->shape[2] << ")\n";

        std::abort();
    }

    const int M = OUT;
    const int K = CIN * KW;
    const int N = COUT;

    std::vector<float> col(
        static_cast<size_t>(M) * K);

    std::vector<float> B(
        static_cast<size_t>(K) * N);

    std::vector<float> tmp(
        static_cast<size_t>(M) * N);

    // -------------------------
    // im2col
    // input NCW
    // -------------------------
    for (int out = 0; out < OUT; ++out) {

        float* dst =
            col.data()
            + static_cast<size_t>(out) * K;

        for (int ic = 0; ic < CIN; ++ic) {

            for (int kw = 0; kw < KW; ++kw) {

                int pos =
                    out * STRIDE + kw - PAD;

                float v = 0.0f;

                if (pos >= 0 && pos < L) {
                    v =
                        input[
                            static_cast<size_t>(ic)
                            * L + pos
                        ];
                }

                dst[ic * KW + kw] = v;
            }
        }
    }

    // -------------------------
    // weight reorder
    // OIW -> [K,N]
    // -------------------------
    for (int ic = 0; ic < CIN; ++ic) {

        for (int kw = 0; kw < KW; ++kw) {

            int k =
                ic * KW + kw;

            for (int oc = 0; oc < COUT; ++oc) {

                B[
                    static_cast<size_t>(k)
                    * N + oc
                ] =
                    weight[
                        (static_cast<size_t>(oc)
                        * CIN + ic)
                        * KW + kw
                    ];
            }
        }
    }

    int64_t shape_a[2] = {M, K};
    int64_t shape_b[2] = {K, N};
    int64_t shape_c[2] = {M, N};

    DLDataType dtype{
        kDLFloat, 32, 1
    };

    DLDevice device{
        kDLCPU, 0
    };

    DLTensor A_t{};
    A_t.data = col.data();
    A_t.device = device;
    A_t.ndim = 2;
    A_t.dtype = dtype;
    A_t.shape = shape_a;

    DLTensor B_t{};
    B_t.data = B.data();
    B_t.device = device;
    B_t.ndim = 2;
    B_t.dtype = dtype;
    B_t.shape = shape_b;

    DLTensor C_t{};
    C_t.data = tmp.data();
    C_t.device = device;
    C_t.ndim = 2;
    C_t.dtype = dtype;
    C_t.shape = shape_c;

    std::vector<const DLTensor*> mm_data{
        &A_t,
        &B_t,
        &C_t
    };

    std::vector<int64_t> shapeA{
        M, K
    };

    std::vector<int64_t> shapeB{
        K, N
    };

    matmul_fp(
        mm_data,
        shapeA,
        shapeB
    );

    // -------------------------
    // [OUT,COUT]
    // -> NCW [1,COUT,OUT]
    // -------------------------
    for (int oc = 0; oc < COUT; ++oc) {

        for (int out = 0; out < OUT; ++out) {

            output[
                static_cast<size_t>(oc)
                * OUT + out
            ] =
                tmp[
                    static_cast<size_t>(out)
                    * COUT + oc
                ];
        }
    }
}
