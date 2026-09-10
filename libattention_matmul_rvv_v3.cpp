#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <dlpack/dlpack.h>
#include <riscv_vector.h>

// Whisper Tiny Attention RVV v3
//
// Input layout (same as v1/v2):
//   A: [BH, M, K]
//   B: [BH, K, N]
//   C: [BH, M, N]
//
// v3 fast paths:
//   QK: M=1, K=64, N=dynamic KV
//   PV: M=1, N=64, K=dynamic KV
//
// Main v3 ideas (inspired by the hierarchical/hybrid RVV MatMul work):
//   1. Keep the v2 Whisper-specific QK/PV fast paths.
//   2. Keep K-loop unroll x4.
//   3. Use TWO m8 accumulators for two adjacent N chunks when possible.
//      This lets one A scalar (Q[k] or P[k]) feed two vector chunks in one K pass.
//   4. Shorten temporary-vector lifetime: Load -> FMA -> reuse temp register.
//      Avoid keeping b0/b1/b2/b3 (or v0/v1/v2/v3) live at the same time.
//   5. Use hierarchical LMUL for the remaining N tail:
//        m8 -> m4 -> m2 -> m1
//      The VLMAX values are queried at runtime, so the code is not hard-coded
//      to VLEN=128.  With VLEN=128 and FP32, VLMAX is 32/16/8/4 for
//      m8/m4/m2/m1 respectively.

namespace {

// -----------------------------------------------------------------------------
// QK helpers: fixed K=64
// -----------------------------------------------------------------------------

static inline void qk_range_m8_k64(
    const float* __restrict__ q,
    const float* __restrict__ kt,
    float* __restrict__ out,
    int N,
    int col,
    int width) {
    const size_t vl = __riscv_vsetvl_e32m8(static_cast<size_t>(width));
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    vfloat32m8_t bv;

    constexpr int K = 64;
    for (int k = 0; k < K; k += 4) {
        const float a0 = q[k + 0];
        bv = __riscv_vle32_v_f32m8(
            kt + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, a0, bv, vl);

        const float a1 = q[k + 1];
        bv = __riscv_vle32_v_f32m8(
            kt + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, a1, bv, vl);

        const float a2 = q[k + 2];
        bv = __riscv_vle32_v_f32m8(
            kt + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, a2, bv, vl);

        const float a3 = q[k + 3];
        bv = __riscv_vle32_v_f32m8(
            kt + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, a3, bv, vl);
    }

    __riscv_vse32_v_f32m8(out + col, acc, vl);
}

static inline void qk_range_m4_k64(
    const float* __restrict__ q,
    const float* __restrict__ kt,
    float* __restrict__ out,
    int N,
    int col,
    int width) {
    const size_t vl = __riscv_vsetvl_e32m4(static_cast<size_t>(width));
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t bv;

    constexpr int K = 64;
    for (int k = 0; k < K; k += 4) {
        const float a0 = q[k + 0];
        bv = __riscv_vle32_v_f32m4(
            kt + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(acc, a0, bv, vl);

        const float a1 = q[k + 1];
        bv = __riscv_vle32_v_f32m4(
            kt + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(acc, a1, bv, vl);

        const float a2 = q[k + 2];
        bv = __riscv_vle32_v_f32m4(
            kt + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(acc, a2, bv, vl);

        const float a3 = q[k + 3];
        bv = __riscv_vle32_v_f32m4(
            kt + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(acc, a3, bv, vl);
    }

    __riscv_vse32_v_f32m4(out + col, acc, vl);
}

static inline void qk_range_m2_k64(
    const float* __restrict__ q,
    const float* __restrict__ kt,
    float* __restrict__ out,
    int N,
    int col,
    int width) {
    const size_t vl = __riscv_vsetvl_e32m2(static_cast<size_t>(width));
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t bv;

    constexpr int K = 64;
    for (int k = 0; k < K; k += 4) {
        const float a0 = q[k + 0];
        bv = __riscv_vle32_v_f32m2(
            kt + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(acc, a0, bv, vl);

        const float a1 = q[k + 1];
        bv = __riscv_vle32_v_f32m2(
            kt + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(acc, a1, bv, vl);

        const float a2 = q[k + 2];
        bv = __riscv_vle32_v_f32m2(
            kt + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(acc, a2, bv, vl);

        const float a3 = q[k + 3];
        bv = __riscv_vle32_v_f32m2(
            kt + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(acc, a3, bv, vl);
    }

    __riscv_vse32_v_f32m2(out + col, acc, vl);
}

static inline void qk_range_m1_k64(
    const float* __restrict__ q,
    const float* __restrict__ kt,
    float* __restrict__ out,
    int N,
    int col,
    int width) {
    const size_t vl = __riscv_vsetvl_e32m1(static_cast<size_t>(width));
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t bv;

    constexpr int K = 64;
    for (int k = 0; k < K; k += 4) {
        const float a0 = q[k + 0];
        bv = __riscv_vle32_v_f32m1(
            kt + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a0, bv, vl);

        const float a1 = q[k + 1];
        bv = __riscv_vle32_v_f32m1(
            kt + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a1, bv, vl);

        const float a2 = q[k + 2];
        bv = __riscv_vle32_v_f32m1(
            kt + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a2, bv, vl);

        const float a3 = q[k + 3];
        bv = __riscv_vle32_v_f32m1(
            kt + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a3, bv, vl);
    }

    __riscv_vse32_v_f32m1(out + col, acc, vl);
}

// Two adjacent m8 chunks in one K pass.
// With VLEN=128/FP32 this computes 64 output columns at a time (32 + 32).
static inline void qk_dual_m8_k64(
    const float* __restrict__ q,
    const float* __restrict__ kt,
    float* __restrict__ out,
    int N,
    int col,
    int chunk_cols) {
    const size_t vl =
        __riscv_vsetvl_e32m8(static_cast<size_t>(chunk_cols));

    vfloat32m8_t acc0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    vfloat32m8_t acc1 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    vfloat32m8_t bv;

    const int col1 = col + static_cast<int>(vl);
    constexpr int K = 64;

    for (int k = 0; k < K; k += 4) {
        const float a0 = q[k + 0];
        const float* row0 = kt + static_cast<size_t>(k + 0) * N;
        bv = __riscv_vle32_v_f32m8(row0 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(acc0, a0, bv, vl);
        bv = __riscv_vle32_v_f32m8(row0 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(acc1, a0, bv, vl);

        const float a1 = q[k + 1];
        const float* row1 = kt + static_cast<size_t>(k + 1) * N;
        bv = __riscv_vle32_v_f32m8(row1 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(acc0, a1, bv, vl);
        bv = __riscv_vle32_v_f32m8(row1 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(acc1, a1, bv, vl);

        const float a2 = q[k + 2];
        const float* row2 = kt + static_cast<size_t>(k + 2) * N;
        bv = __riscv_vle32_v_f32m8(row2 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(acc0, a2, bv, vl);
        bv = __riscv_vle32_v_f32m8(row2 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(acc1, a2, bv, vl);

        const float a3 = q[k + 3];
        const float* row3 = kt + static_cast<size_t>(k + 3) * N;
        bv = __riscv_vle32_v_f32m8(row3 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(acc0, a3, bv, vl);
        bv = __riscv_vle32_v_f32m8(row3 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(acc1, a3, bv, vl);
    }

    __riscv_vse32_v_f32m8(out + col, acc0, vl);
    __riscv_vse32_v_f32m8(out + col1, acc1, vl);
}

static inline void qk_hierarchical_tail_k64(
    const float* __restrict__ q,
    const float* __restrict__ kt,
    float* __restrict__ out,
    int N,
    int start_col) {
    int col = start_col;

    const int vlmax8 = static_cast<int>(__riscv_vsetvlmax_e32m8());
    const int vlmax4 = static_cast<int>(__riscv_vsetvlmax_e32m4());
    const int vlmax2 = static_cast<int>(__riscv_vsetvlmax_e32m2());
    const int vlmax1 = static_cast<int>(__riscv_vsetvlmax_e32m1());

    while (N - col >= vlmax8) {
        qk_range_m8_k64(q, kt, out, N, col, vlmax8);
        col += vlmax8;
    }

    if (N - col >= vlmax4) {
        qk_range_m4_k64(q, kt, out, N, col, vlmax4);
        col += vlmax4;
    }

    if (N - col >= vlmax2) {
        qk_range_m2_k64(q, kt, out, N, col, vlmax2);
        col += vlmax2;
    }

    // Includes both a full m1 chunk and the final partial 1..(VLMAX-1) tail.
    while (col < N) {
        const int width = (N - col >= vlmax1) ? vlmax1 : (N - col);
        qk_range_m1_k64(q, kt, out, N, col, width);
        col += width;
    }
}

__attribute__((noinline))
static void whisper_qk_m1_k64_rvv_v3(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BH,
    int N) {
    constexpr int K = 64;

    const int vlmax8 = static_cast<int>(__riscv_vsetvlmax_e32m8());
    const int dual_cols = 2 * vlmax8;

    for (int b = 0; b < BH; ++b) {
        const float* __restrict__ q = A + static_cast<size_t>(b) * K;
        const float* __restrict__ kt =
            B + static_cast<size_t>(b) * K * N;
        float* __restrict__ out = C + static_cast<size_t>(b) * N;

        int col = 0;

        // Main region: two full m8 chunks per K pass.
        while (N - col >= dual_cols) {
            qk_dual_m8_k64(q, kt, out, N, col, vlmax8);
            col += dual_cols;
        }

        // Tail: m8 -> m4 -> m2 -> m1.
        if (col < N) {
            qk_hierarchical_tail_k64(q, kt, out, N, col);
        }
    }
}

// -----------------------------------------------------------------------------
// PV helpers: fixed N=64, dynamic K
// -----------------------------------------------------------------------------

static inline void pv_range_m8(
    const float* __restrict__ p,
    const float* __restrict__ v,
    float* __restrict__ out,
    int K,
    int N,
    int col,
    int width) {
    const size_t vl = __riscv_vsetvl_e32m8(static_cast<size_t>(width));
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    vfloat32m8_t vv;

    int k = 0;
    for (; k + 3 < K; k += 4) {
        const float a0 = p[k + 0];
        vv = __riscv_vle32_v_f32m8(
            v + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, a0, vv, vl);

        const float a1 = p[k + 1];
        vv = __riscv_vle32_v_f32m8(
            v + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, a1, vv, vl);

        const float a2 = p[k + 2];
        vv = __riscv_vle32_v_f32m8(
            v + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, a2, vv, vl);

        const float a3 = p[k + 3];
        vv = __riscv_vle32_v_f32m8(
            v + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, a3, vv, vl);
    }

    for (; k < K; ++k) {
        vv = __riscv_vle32_v_f32m8(
            v + static_cast<size_t>(k) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, p[k], vv, vl);
    }

    __riscv_vse32_v_f32m8(out + col, acc, vl);
}

static inline void pv_range_m4(
    const float* __restrict__ p,
    const float* __restrict__ v,
    float* __restrict__ out,
    int K,
    int N,
    int col,
    int width) {
    const size_t vl = __riscv_vsetvl_e32m4(static_cast<size_t>(width));
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vv;

    int k = 0;
    for (; k + 3 < K; k += 4) {
        const float a0 = p[k + 0];
        vv = __riscv_vle32_v_f32m4(
            v + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(acc, a0, vv, vl);

        const float a1 = p[k + 1];
        vv = __riscv_vle32_v_f32m4(
            v + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(acc, a1, vv, vl);

        const float a2 = p[k + 2];
        vv = __riscv_vle32_v_f32m4(
            v + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(acc, a2, vv, vl);

        const float a3 = p[k + 3];
        vv = __riscv_vle32_v_f32m4(
            v + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(acc, a3, vv, vl);
    }

    for (; k < K; ++k) {
        vv = __riscv_vle32_v_f32m4(
            v + static_cast<size_t>(k) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(acc, p[k], vv, vl);
    }

    __riscv_vse32_v_f32m4(out + col, acc, vl);
}

static inline void pv_range_m2(
    const float* __restrict__ p,
    const float* __restrict__ v,
    float* __restrict__ out,
    int K,
    int N,
    int col,
    int width) {
    const size_t vl = __riscv_vsetvl_e32m2(static_cast<size_t>(width));
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t vv;

    int k = 0;
    for (; k + 3 < K; k += 4) {
        const float a0 = p[k + 0];
        vv = __riscv_vle32_v_f32m2(
            v + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(acc, a0, vv, vl);

        const float a1 = p[k + 1];
        vv = __riscv_vle32_v_f32m2(
            v + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(acc, a1, vv, vl);

        const float a2 = p[k + 2];
        vv = __riscv_vle32_v_f32m2(
            v + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(acc, a2, vv, vl);

        const float a3 = p[k + 3];
        vv = __riscv_vle32_v_f32m2(
            v + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(acc, a3, vv, vl);
    }

    for (; k < K; ++k) {
        vv = __riscv_vle32_v_f32m2(
            v + static_cast<size_t>(k) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(acc, p[k], vv, vl);
    }

    __riscv_vse32_v_f32m2(out + col, acc, vl);
}

static inline void pv_range_m1(
    const float* __restrict__ p,
    const float* __restrict__ v,
    float* __restrict__ out,
    int K,
    int N,
    int col,
    int width) {
    const size_t vl = __riscv_vsetvl_e32m1(static_cast<size_t>(width));
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vv;

    int k = 0;
    for (; k + 3 < K; k += 4) {
        const float a0 = p[k + 0];
        vv = __riscv_vle32_v_f32m1(
            v + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a0, vv, vl);

        const float a1 = p[k + 1];
        vv = __riscv_vle32_v_f32m1(
            v + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a1, vv, vl);

        const float a2 = p[k + 2];
        vv = __riscv_vle32_v_f32m1(
            v + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a2, vv, vl);

        const float a3 = p[k + 3];
        vv = __riscv_vle32_v_f32m1(
            v + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a3, vv, vl);
    }

    for (; k < K; ++k) {
        vv = __riscv_vle32_v_f32m1(
            v + static_cast<size_t>(k) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, p[k], vv, vl);
    }

    __riscv_vse32_v_f32m1(out + col, acc, vl);
}

static inline void pv_dual_m8(
    const float* __restrict__ p,
    const float* __restrict__ v,
    float* __restrict__ out,
    int K,
    int N,
    int col,
    int chunk_cols) {
    const size_t vl =
        __riscv_vsetvl_e32m8(static_cast<size_t>(chunk_cols));

    vfloat32m8_t acc0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    vfloat32m8_t acc1 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    vfloat32m8_t vv;

    const int col1 = col + static_cast<int>(vl);

    int k = 0;
    for (; k + 3 < K; k += 4) {
        const float a0 = p[k + 0];
        const float* row0 = v + static_cast<size_t>(k + 0) * N;
        vv = __riscv_vle32_v_f32m8(row0 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(acc0, a0, vv, vl);
        vv = __riscv_vle32_v_f32m8(row0 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(acc1, a0, vv, vl);

        const float a1 = p[k + 1];
        const float* row1 = v + static_cast<size_t>(k + 1) * N;
        vv = __riscv_vle32_v_f32m8(row1 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(acc0, a1, vv, vl);
        vv = __riscv_vle32_v_f32m8(row1 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(acc1, a1, vv, vl);

        const float a2 = p[k + 2];
        const float* row2 = v + static_cast<size_t>(k + 2) * N;
        vv = __riscv_vle32_v_f32m8(row2 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(acc0, a2, vv, vl);
        vv = __riscv_vle32_v_f32m8(row2 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(acc1, a2, vv, vl);

        const float a3 = p[k + 3];
        const float* row3 = v + static_cast<size_t>(k + 3) * N;
        vv = __riscv_vle32_v_f32m8(row3 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(acc0, a3, vv, vl);
        vv = __riscv_vle32_v_f32m8(row3 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(acc1, a3, vv, vl);
    }

    for (; k < K; ++k) {
        const float a = p[k];
        const float* row = v + static_cast<size_t>(k) * N;
        vv = __riscv_vle32_v_f32m8(row + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(acc0, a, vv, vl);
        vv = __riscv_vle32_v_f32m8(row + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(acc1, a, vv, vl);
    }

    __riscv_vse32_v_f32m8(out + col, acc0, vl);
    __riscv_vse32_v_f32m8(out + col1, acc1, vl);
}

static inline void pv_hierarchical_tail(
    const float* __restrict__ p,
    const float* __restrict__ v,
    float* __restrict__ out,
    int K,
    int N,
    int start_col) {
    int col = start_col;

    const int vlmax8 = static_cast<int>(__riscv_vsetvlmax_e32m8());
    const int vlmax4 = static_cast<int>(__riscv_vsetvlmax_e32m4());
    const int vlmax2 = static_cast<int>(__riscv_vsetvlmax_e32m2());
    const int vlmax1 = static_cast<int>(__riscv_vsetvlmax_e32m1());

    while (N - col >= vlmax8) {
        pv_range_m8(p, v, out, K, N, col, vlmax8);
        col += vlmax8;
    }

    if (N - col >= vlmax4) {
        pv_range_m4(p, v, out, K, N, col, vlmax4);
        col += vlmax4;
    }

    if (N - col >= vlmax2) {
        pv_range_m2(p, v, out, K, N, col, vlmax2);
        col += vlmax2;
    }

    while (col < N) {
        const int width = (N - col >= vlmax1) ? vlmax1 : (N - col);
        pv_range_m1(p, v, out, K, N, col, width);
        col += width;
    }
}

__attribute__((noinline))
static void whisper_pv_m1_n64_rvv_v3(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BH,
    int K) {
    constexpr int N = 64;

    const int vlmax8 = static_cast<int>(__riscv_vsetvlmax_e32m8());
    const int dual_cols = 2 * vlmax8;

    for (int b = 0; b < BH; ++b) {
        const float* __restrict__ p = A + static_cast<size_t>(b) * K;
        const float* __restrict__ v =
            B + static_cast<size_t>(b) * K * N;
        float* __restrict__ out = C + static_cast<size_t>(b) * N;

        int col = 0;

        // On VLEN=128/FP32: dual_cols = 64, so PV's whole N=64 output is
        // produced with two m8 accumulators in a single K traversal.
        while (N - col >= dual_cols) {
            pv_dual_m8(p, v, out, K, N, col, vlmax8);
            col += dual_cols;
        }

        // Portability path for other VLENs, or any remaining columns.
        if (col < N) {
            pv_hierarchical_tail(p, v, out, K, N, col);
        }
    }
}

// -----------------------------------------------------------------------------
// Generic fallback (kept close to v2 for controlled comparison)
// -----------------------------------------------------------------------------

static inline void attention_bmm_generic_rvv_m4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BH,
    int M,
    int K,
    int N) {
    for (int b = 0; b < BH; ++b) {
        const float* __restrict__ Ab =
            A + static_cast<size_t>(b) * M * K;
        const float* __restrict__ Bb =
            B + static_cast<size_t>(b) * K * N;
        float* __restrict__ Cb =
            C + static_cast<size_t>(b) * M * N;

        for (int i = 0; i < M; ++i) {
            const float* __restrict__ arow =
                Ab + static_cast<size_t>(i) * K;
            float* __restrict__ crow =
                Cb + static_cast<size_t>(i) * N;

            int col = 0;
            while (col < N) {
                const size_t vl =
                    __riscv_vsetvl_e32m4(static_cast<size_t>(N - col));
                vfloat32m4_t acc =
                    __riscv_vfmv_v_f_f32m4(0.0f, vl);

                int k = 0;
                for (; k + 3 < K; k += 4) {
                    vfloat32m4_t b0 = __riscv_vle32_v_f32m4(
                        Bb + static_cast<size_t>(k + 0) * N + col, vl);
                    vfloat32m4_t b1 = __riscv_vle32_v_f32m4(
                        Bb + static_cast<size_t>(k + 1) * N + col, vl);
                    vfloat32m4_t b2 = __riscv_vle32_v_f32m4(
                        Bb + static_cast<size_t>(k + 2) * N + col, vl);
                    vfloat32m4_t b3 = __riscv_vle32_v_f32m4(
                        Bb + static_cast<size_t>(k + 3) * N + col, vl);

                    acc = __riscv_vfmacc_vf_f32m4(
                        acc, arow[k + 0], b0, vl);
                    acc = __riscv_vfmacc_vf_f32m4(
                        acc, arow[k + 1], b1, vl);
                    acc = __riscv_vfmacc_vf_f32m4(
                        acc, arow[k + 2], b2, vl);
                    acc = __riscv_vfmacc_vf_f32m4(
                        acc, arow[k + 3], b3, vl);
                }

                for (; k < K; ++k) {
                    const vfloat32m4_t bv = __riscv_vle32_v_f32m4(
                        Bb + static_cast<size_t>(k) * N + col, vl);
                    acc = __riscv_vfmacc_vf_f32m4(
                        acc, arow[k], bv, vl);
                }

                __riscv_vse32_v_f32m4(crow + col, acc, vl);
                col += static_cast<int>(vl);
            }
        }
    }
}

}  // namespace

extern "C" void attention_bmm_rvv_f32(
    const float* A,
    const float* B,
    float* C,
    int BH,
    int M,
    int K,
    int N) {
    // Whisper decoder QK: [BH,1,64] x [BH,64,KV]
    if (M == 1 && K == 64) {
        whisper_qk_m1_k64_rvv_v3(A, B, C, BH, N);
        return;
    }

    // Whisper decoder PV: [BH,1,KV] x [BH,KV,64]
    if (M == 1 && N == 64) {
        whisper_pv_m1_n64_rvv_v3(A, B, C, BH, K);
        return;
    }

    // Encoder / other Attention shapes.
    attention_bmm_generic_rvv_m4(A, B, C, BH, M, K, N);
}

extern "C" void attention_matmul(
    std::vector<const DLTensor*>& data_entry_,
    std::vector<int64_t>& shapeA,
    std::vector<int64_t>& shapeB) {
    (void)shapeA;
    (void)shapeB;

    if (data_entry_.size() < 3) {
        std::cerr << "[ATTENTION_RVV_V3] expected A, B, C\n";
        return;
    }

    const DLTensor* A_t = data_entry_[0];
    const DLTensor* B_t = data_entry_[1];
    const DLTensor* C_t = data_entry_[2];

    if (A_t->ndim != 3 || B_t->ndim != 3 || C_t->ndim != 3) {
        std::cerr << "[ATTENTION_RVV_V3] only 3D BMM supported\n";
        return;
    }

    const int BH = static_cast<int>(A_t->shape[0]);
    const int M = static_cast<int>(A_t->shape[1]);
    const int K = static_cast<int>(A_t->shape[2]);

    const int B_BH = static_cast<int>(B_t->shape[0]);
    const int B_K = static_cast<int>(B_t->shape[1]);
    const int N = static_cast<int>(B_t->shape[2]);

    if (B_BH != BH || B_K != K ||
        static_cast<int>(C_t->shape[0]) != BH ||
        static_cast<int>(C_t->shape[1]) != M ||
        static_cast<int>(C_t->shape[2]) != N) {
        std::cerr << "[ATTENTION_RVV_V3] shape mismatch\n";
        return;
    }

    const float* A = static_cast<const float*>(A_t->data);
    const float* B = static_cast<const float*>(B_t->data);
    float* C = static_cast<float*>(C_t->data);

    attention_bmm_rvv_f32(A, B, C, BH, M, K, N);
}
