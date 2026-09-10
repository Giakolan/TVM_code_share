#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>
#include <dlpack/dlpack.h>
#include <riscv_vector.h>

// Whisper Tiny Attention RVV v2
// A:[BH,M,K], B:[BH,K,N], C:[BH,M,N]
// Fast paths:
//   QK: M=1, K=64, N=dynamic KV
//   PV: M=1, N=64, K=dynamic KV

static inline void whisper_qk_m1_k64_rvv_m8(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BH,
    int N
) {
    constexpr int K = 64;

    for (int b = 0; b < BH; ++b) {
        const float* __restrict__ q = A + (size_t)b * K;
        const float* __restrict__ kt = B + (size_t)b * K * N;
        float* __restrict__ out = C + (size_t)b * N;

        int col = 0;
        while (col < N) {
            size_t vl = __riscv_vsetvl_e32m8((size_t)(N - col));
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);

            for (int k = 0; k < K; k += 4) {
                vfloat32m8_t b0 =
                    __riscv_vle32_v_f32m8(kt + (size_t)(k + 0) * N + col, vl);
                vfloat32m8_t b1 =
                    __riscv_vle32_v_f32m8(kt + (size_t)(k + 1) * N + col, vl);
                vfloat32m8_t b2 =
                    __riscv_vle32_v_f32m8(kt + (size_t)(k + 2) * N + col, vl);
                vfloat32m8_t b3 =
                    __riscv_vle32_v_f32m8(kt + (size_t)(k + 3) * N + col, vl);

                acc = __riscv_vfmacc_vf_f32m8(acc, q[k + 0], b0, vl);
                acc = __riscv_vfmacc_vf_f32m8(acc, q[k + 1], b1, vl);
                acc = __riscv_vfmacc_vf_f32m8(acc, q[k + 2], b2, vl);
                acc = __riscv_vfmacc_vf_f32m8(acc, q[k + 3], b3, vl);
            }

            __riscv_vse32_v_f32m8(out + col, acc, vl);
            col += (int)vl;
        }
    }
}

static inline void whisper_pv_m1_n64_rvv_m8(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BH,
    int K
) {
    constexpr int N = 64;

    for (int b = 0; b < BH; ++b) {
        const float* __restrict__ p = A + (size_t)b * K;
        const float* __restrict__ v = B + (size_t)b * K * N;
        float* __restrict__ out = C + (size_t)b * N;

        int col = 0;
        while (col < N) {
            size_t vl = __riscv_vsetvl_e32m8((size_t)(N - col));
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);

            int k = 0;
            for (; k + 3 < K; k += 4) {
                vfloat32m8_t v0 =
                    __riscv_vle32_v_f32m8(v + (size_t)(k + 0) * N + col, vl);
                vfloat32m8_t v1 =
                    __riscv_vle32_v_f32m8(v + (size_t)(k + 1) * N + col, vl);
                vfloat32m8_t v2 =
                    __riscv_vle32_v_f32m8(v + (size_t)(k + 2) * N + col, vl);
                vfloat32m8_t v3 =
                    __riscv_vle32_v_f32m8(v + (size_t)(k + 3) * N + col, vl);

                acc = __riscv_vfmacc_vf_f32m8(acc, p[k + 0], v0, vl);
                acc = __riscv_vfmacc_vf_f32m8(acc, p[k + 1], v1, vl);
                acc = __riscv_vfmacc_vf_f32m8(acc, p[k + 2], v2, vl);
                acc = __riscv_vfmacc_vf_f32m8(acc, p[k + 3], v3, vl);
            }

            for (; k < K; ++k) {
                vfloat32m8_t vv =
                    __riscv_vle32_v_f32m8(v + (size_t)k * N + col, vl);
                acc = __riscv_vfmacc_vf_f32m8(acc, p[k], vv, vl);
            }

            __riscv_vse32_v_f32m8(out + col, acc, vl);
            col += (int)vl;
        }
    }
}

static inline void attention_bmm_generic_rvv_m4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BH,
    int M,
    int K,
    int N
) {
    for (int b = 0; b < BH; ++b) {
        const float* __restrict__ Ab = A + (size_t)b * M * K;
        const float* __restrict__ Bb = B + (size_t)b * K * N;
        float* __restrict__ Cb = C + (size_t)b * M * N;

        for (int i = 0; i < M; ++i) {
            const float* __restrict__ arow = Ab + (size_t)i * K;
            float* __restrict__ crow = Cb + (size_t)i * N;

            int col = 0;
            while (col < N) {
                size_t vl = __riscv_vsetvl_e32m4((size_t)(N - col));
                vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

                int k = 0;
                for (; k + 3 < K; k += 4) {
                    vfloat32m4_t b0 =
                        __riscv_vle32_v_f32m4(Bb + (size_t)(k + 0) * N + col, vl);
                    vfloat32m4_t b1 =
                        __riscv_vle32_v_f32m4(Bb + (size_t)(k + 1) * N + col, vl);
                    vfloat32m4_t b2 =
                        __riscv_vle32_v_f32m4(Bb + (size_t)(k + 2) * N + col, vl);
                    vfloat32m4_t b3 =
                        __riscv_vle32_v_f32m4(Bb + (size_t)(k + 3) * N + col, vl);

                    acc = __riscv_vfmacc_vf_f32m4(acc, arow[k + 0], b0, vl);
                    acc = __riscv_vfmacc_vf_f32m4(acc, arow[k + 1], b1, vl);
                    acc = __riscv_vfmacc_vf_f32m4(acc, arow[k + 2], b2, vl);
                    acc = __riscv_vfmacc_vf_f32m4(acc, arow[k + 3], b3, vl);
                }

                for (; k < K; ++k) {
                    vfloat32m4_t bv =
                        __riscv_vle32_v_f32m4(Bb + (size_t)k * N + col, vl);
                    acc = __riscv_vfmacc_vf_f32m4(acc, arow[k], bv, vl);
                }

                __riscv_vse32_v_f32m4(crow + col, acc, vl);
                col += (int)vl;
            }
        }
    }
}

extern "C"
void attention_bmm_rvv_f32(
    const float* A,
    const float* B,
    float* C,
    int BH,
    int M,
    int K,
    int N
) {
    // Whisper decoder QK: [BH,1,64] x [BH,64,KV]
    if (M == 1 && K == 64) {
        whisper_qk_m1_k64_rvv_m8(A, B, C, BH, N);
        return;
    }

    // Whisper decoder PV: [BH,1,KV] x [BH,KV,64]
    if (M == 1 && N == 64) {
        whisper_pv_m1_n64_rvv_m8(A, B, C, BH, K);
        return;
    }

    // Encoder / other attention shapes.
    attention_bmm_generic_rvv_m4(A, B, C, BH, M, K, N);
}

extern "C"
void attention_matmul(
    std::vector<const DLTensor*>& data_entry_,
    std::vector<int64_t>& shapeA,
    std::vector<int64_t>& shapeB
) {
    (void)shapeA;
    (void)shapeB;

    if (data_entry_.size() < 3) {
        std::cerr << "[ATTENTION_RVV_V2] expected A, B, C\n";
        return;
    }

    const DLTensor* A_t = data_entry_[0];
    const DLTensor* B_t = data_entry_[1];
    const DLTensor* C_t = data_entry_[2];

    if (A_t->ndim != 3 || B_t->ndim != 3 || C_t->ndim != 3) {
        std::cerr << "[ATTENTION_RVV_V2] only 3D BMM supported\n";
        return;
    }

    int BH = (int)A_t->shape[0];
    int M  = (int)A_t->shape[1];
    int K  = (int)A_t->shape[2];

    int B_BH = (int)B_t->shape[0];
    int B_K  = (int)B_t->shape[1];
    int N    = (int)B_t->shape[2];

    if (B_BH != BH || B_K != K ||
        (int)C_t->shape[0] != BH ||
        (int)C_t->shape[1] != M ||
        (int)C_t->shape[2] != N) {
        std::cerr << "[ATTENTION_RVV_V2] shape mismatch\n";
        return;
    }

    const float* A = static_cast<const float*>(A_t->data);
    const float* B = static_cast<const float*>(B_t->data);
    float* C = static_cast<float*>(C_t->data);

    attention_bmm_rvv_f32(A, B, C, BH, M, K, N);
}
