/*
 * libattention_matmul_rvv_v4_compatible.cpp
 *
 * Whisper Attention RVV -- v3 + v4 compatible version
 *
 * Goal
 * ----
 * Keep v3's ability to execute normal/dynamic Whisper Attention shapes,
 * while adding a stronger fast path for the important Cross-Attention
 * N=1500 case.
 *
 * Supported BMM form
 * ------------------
 *   A: [BH, M, K]
 *   B: [BH, K, N]
 *   C: [BH, M, N]
 *
 * Main dispatch
 * -------------
 *   1) K=64, N=1500
 *      -> v4 N=1500 specialized QK kernel
 *         - BH is NOT fixed to 6
 *         - M is NOT fixed to 1
 *         - every [1,64] row uses the N=1500 specialized row kernel
 *
 *   2) M=1, K=64, N=dynamic
 *      -> v3 dynamic-QK fast path
 *
 *   3) M=1, N=64, K=dynamic
 *      -> v3 dynamic-PV fast path
 *
 *   4) all other Attention BMM shapes
 *      -> generic RVV m4 fallback
 *
 * Important difference from the previous v4
 * -----------------------------------------
 * This version does NOT assume that Attention itself is only
 * [6,1,64] x [6,64,1500].
 *
 * N=1500 is only an EXTRA fast path.  If the real model presents another
 * KV length, PV, encoder-attention shape, a different BH, or M>1, the
 * operator still has a valid v3/generic execution path.
 *
 * Integration-oriented choices
 * ----------------------------
 * - No pthread / std::thread dependency in this file.
 * - No hard-coded BH=6 requirement.
 * - N=1500 fast path supports arbitrary BH and arbitrary M when K=64.
 * - Keeps the same public symbols:
 *       attention_bmm_rvv_f32(...)
 *       attention_matmul(...)
 *
 * DLTensor entry additionally supports:
 *   [BH,M,K] x [BH,K,N]
 *   [BH,M,K] x [K,N]       (shared B)
 *   [M,K]    x [K,N]
 *
 * FP32 contiguous tensors only.
 */

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <dlpack/dlpack.h>
#include <riscv_vector.h>

namespace {

// ============================================================================
// Tensor helpers
// ============================================================================

static inline bool is_fp32(const DLTensor* t) {
    return t != nullptr &&
           t->dtype.code == kDLFloat &&
           t->dtype.bits == 32 &&
           t->dtype.lanes == 1;
}

static inline bool is_compact_contiguous(const DLTensor* t) {
    if (t == nullptr) return false;
    if (t->strides == nullptr) return true;

    int64_t expected = 1;
    for (int i = t->ndim - 1; i >= 0; --i) {
        if (t->shape[i] > 1 && t->strides[i] != expected) {
            return false;
        }
        expected *= t->shape[i];
    }
    return true;
}

static inline const float* tensor_data_const(const DLTensor* t) {
    const auto* base =
        reinterpret_cast<const std::uint8_t*>(t->data) + t->byte_offset;
    return reinterpret_cast<const float*>(base);
}

static inline float* tensor_data_mutable(const DLTensor* t) {
    auto* base =
        reinterpret_cast<std::uint8_t*>(t->data) + t->byte_offset;
    return reinterpret_cast<float*>(base);
}

// ============================================================================
// QK helpers -- fixed K=64
// ============================================================================
//
// Whisper head dimension is 64 in the target configuration.
//
// B is already in the BMM-friendly layout:
//   B = [BH, 64, N]
// so B[k][col : col+vl] is contiguous and can use unit-stride vle32.
//
// v3 techniques retained:
//   - LMUL=m8 main path
//   - two m8 accumulators
//   - K unroll x4
//   - Load -> FMA -> reuse the temporary vector
//   - hierarchical m8 -> m4 -> m2 -> m1 tail
// ============================================================================

static inline void qk_range_m8_k64(
    const float* __restrict__ q,
    const float* __restrict__ kt,
    float* __restrict__ out,
    int N,
    int col,
    int width) {

    const size_t vl =
        __riscv_vsetvl_e32m8(static_cast<size_t>(width));

    vfloat32m8_t acc =
        __riscv_vfmv_v_f_f32m8(0.0f, vl);

    vfloat32m8_t bv;

    for (int k = 0; k < 64; k += 4) {
        bv = __riscv_vle32_v_f32m8(
            kt + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(
            acc, q[k + 0], bv, vl);

        bv = __riscv_vle32_v_f32m8(
            kt + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(
            acc, q[k + 1], bv, vl);

        bv = __riscv_vle32_v_f32m8(
            kt + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(
            acc, q[k + 2], bv, vl);

        bv = __riscv_vle32_v_f32m8(
            kt + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(
            acc, q[k + 3], bv, vl);
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

    const size_t vl =
        __riscv_vsetvl_e32m4(static_cast<size_t>(width));

    vfloat32m4_t acc =
        __riscv_vfmv_v_f_f32m4(0.0f, vl);

    vfloat32m4_t bv;

    for (int k = 0; k < 64; k += 4) {
        bv = __riscv_vle32_v_f32m4(
            kt + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(
            acc, q[k + 0], bv, vl);

        bv = __riscv_vle32_v_f32m4(
            kt + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(
            acc, q[k + 1], bv, vl);

        bv = __riscv_vle32_v_f32m4(
            kt + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(
            acc, q[k + 2], bv, vl);

        bv = __riscv_vle32_v_f32m4(
            kt + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(
            acc, q[k + 3], bv, vl);
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

    const size_t vl =
        __riscv_vsetvl_e32m2(static_cast<size_t>(width));

    vfloat32m2_t acc =
        __riscv_vfmv_v_f_f32m2(0.0f, vl);

    vfloat32m2_t bv;

    for (int k = 0; k < 64; k += 4) {
        bv = __riscv_vle32_v_f32m2(
            kt + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(
            acc, q[k + 0], bv, vl);

        bv = __riscv_vle32_v_f32m2(
            kt + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(
            acc, q[k + 1], bv, vl);

        bv = __riscv_vle32_v_f32m2(
            kt + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(
            acc, q[k + 2], bv, vl);

        bv = __riscv_vle32_v_f32m2(
            kt + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(
            acc, q[k + 3], bv, vl);
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

    const size_t vl =
        __riscv_vsetvl_e32m1(static_cast<size_t>(width));

    vfloat32m1_t acc =
        __riscv_vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t bv;

    for (int k = 0; k < 64; k += 4) {
        bv = __riscv_vle32_v_f32m1(
            kt + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(
            acc, q[k + 0], bv, vl);

        bv = __riscv_vle32_v_f32m1(
            kt + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(
            acc, q[k + 1], bv, vl);

        bv = __riscv_vle32_v_f32m1(
            kt + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(
            acc, q[k + 2], bv, vl);

        bv = __riscv_vle32_v_f32m1(
            kt + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(
            acc, q[k + 3], bv, vl);
    }

    __riscv_vse32_v_f32m1(out + col, acc, vl);
}


// Two adjacent m8 chunks in ONE K traversal.
//
// VLEN=128 + FP32:
//   VLMAX(m8) = 32
//   acc0 = 32 columns
//   acc1 = 32 columns
//   total = 64 columns per K traversal.
static inline void qk_dual_m8_k64(
    const float* __restrict__ q,
    const float* __restrict__ kt,
    float* __restrict__ out,
    int N,
    int col,
    int chunk_cols) {

    const size_t vl =
        __riscv_vsetvl_e32m8(static_cast<size_t>(chunk_cols));

    vfloat32m8_t acc0 =
        __riscv_vfmv_v_f_f32m8(0.0f, vl);

    vfloat32m8_t acc1 =
        __riscv_vfmv_v_f_f32m8(0.0f, vl);

    vfloat32m8_t bv;

    const int col1 = col + static_cast<int>(vl);

    for (int k = 0; k < 64; k += 4) {
        const float* row0 =
            kt + static_cast<size_t>(k + 0) * N;

        bv = __riscv_vle32_v_f32m8(row0 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(
            acc0, q[k + 0], bv, vl);

        bv = __riscv_vle32_v_f32m8(row0 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(
            acc1, q[k + 0], bv, vl);


        const float* row1 =
            kt + static_cast<size_t>(k + 1) * N;

        bv = __riscv_vle32_v_f32m8(row1 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(
            acc0, q[k + 1], bv, vl);

        bv = __riscv_vle32_v_f32m8(row1 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(
            acc1, q[k + 1], bv, vl);


        const float* row2 =
            kt + static_cast<size_t>(k + 2) * N;

        bv = __riscv_vle32_v_f32m8(row2 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(
            acc0, q[k + 2], bv, vl);

        bv = __riscv_vle32_v_f32m8(row2 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(
            acc1, q[k + 2], bv, vl);


        const float* row3 =
            kt + static_cast<size_t>(k + 3) * N;

        bv = __riscv_vle32_v_f32m8(row3 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(
            acc0, q[k + 3], bv, vl);

        bv = __riscv_vle32_v_f32m8(row3 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(
            acc1, q[k + 3], bv, vl);
    }

    __riscv_vse32_v_f32m8(out + col,  acc0, vl);
    __riscv_vse32_v_f32m8(out + col1, acc1, vl);
}


static inline void qk_hierarchical_tail_k64(
    const float* __restrict__ q,
    const float* __restrict__ kt,
    float* __restrict__ out,
    int N,
    int start_col) {

    int col = start_col;

    const int vlmax8 =
        static_cast<int>(__riscv_vsetvlmax_e32m8());

    const int vlmax4 =
        static_cast<int>(__riscv_vsetvlmax_e32m4());

    const int vlmax2 =
        static_cast<int>(__riscv_vsetvlmax_e32m2());

    const int vlmax1 =
        static_cast<int>(__riscv_vsetvlmax_e32m1());

    while (N - col >= vlmax8) {
        qk_range_m8_k64(
            q, kt, out, N, col, vlmax8);
        col += vlmax8;
    }

    if (N - col >= vlmax4) {
        qk_range_m4_k64(
            q, kt, out, N, col, vlmax4);
        col += vlmax4;
    }

    if (N - col >= vlmax2) {
        qk_range_m2_k64(
            q, kt, out, N, col, vlmax2);
        col += vlmax2;
    }

    while (col < N) {
        const int width =
            (N - col >= vlmax1)
                ? vlmax1
                : (N - col);

        qk_range_m1_k64(
            q, kt, out, N, col, width);

        col += width;
    }
}


// ============================================================================
// v4 emphasized path -- K=64, N=1500
// ============================================================================
//
// This is deliberately only a ROW kernel.
//
// It computes:
//
//   [1,64] x [64,1500] -> [1,1500]
//
// The higher-level dispatcher loops over ANY BH and ANY M.
//
// Therefore these are all valid:
//   [6,1,64]   x [6,64,1500]
//   [6,10,64]  x [6,64,1500]
//   [4,1,64]   x [4,64,1500]
//   [1,1,64]   x [1,64,1500]
//
// On VLEN=128:
//   m8 = 32 FP32
//   dual-m8 = 64 columns
//
//   1500 = 23 * 64 + 28
//        = 1472 + 16 + 8 + 4
//
// So the intended fast path is:
//   23 x dual-m8
//   + m4(16)
//   + m2(8)
//   + m1(4)
//
// On another VLEN, correctness is preserved through a dynamic RVV fallback.
// ============================================================================

constexpr int kQK_K = 64;
constexpr int kCrossN = 1500;

__attribute__((noinline))
static void qk_k64_n1500_row_v4(
    const float* __restrict__ q,
    const float* __restrict__ kt,
    float* __restrict__ out) {

    const int vlmax8 =
        static_cast<int>(__riscv_vsetvlmax_e32m8());

    const int vlmax4 =
        static_cast<int>(__riscv_vsetvlmax_e32m4());

    const int vlmax2 =
        static_cast<int>(__riscv_vsetvlmax_e32m2());

    const int vlmax1 =
        static_cast<int>(__riscv_vsetvlmax_e32m1());

    // Best-known target configuration used in the project:
    // VLEN=128, SEW=32.
    if (vlmax8 == 32 &&
        vlmax4 == 16 &&
        vlmax2 == 8 &&
        vlmax1 == 4) {

        // 23 x 64 = 1472 columns.
        for (int block = 0; block < 23; ++block) {
            qk_dual_m8_k64(
                q,
                kt,
                out,
                kCrossN,
                block * 64,
                32
            );
        }

        // Fixed 28-element tail.
        qk_range_m4_k64(
            q, kt, out, kCrossN, 1472, 16);

        qk_range_m2_k64(
            q, kt, out, kCrossN, 1488, 8);

        qk_range_m1_k64(
            q, kt, out, kCrossN, 1496, 4);

        return;
    }

    // Portable v4 fallback for a different hardware VLEN.
    int col = 0;
    const int dual_cols = 2 * vlmax8;

    while (kCrossN - col >= dual_cols) {
        qk_dual_m8_k64(
            q,
            kt,
            out,
            kCrossN,
            col,
            vlmax8
        );

        col += dual_cols;
    }

    if (col < kCrossN) {
        qk_hierarchical_tail_k64(
            q,
            kt,
            out,
            kCrossN,
            col
        );
    }
}


// N=1500 optimization is NOT restricted to BH=6 or M=1.
static inline void whisper_qk_k64_n1500_v4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BH,
    int M) {

    for (int b = 0; b < BH; ++b) {
        const float* __restrict__ Ab =
            A + static_cast<size_t>(b) * M * kQK_K;

        const float* __restrict__ Bb =
            B + static_cast<size_t>(b) * kQK_K * kCrossN;

        float* __restrict__ Cb =
            C + static_cast<size_t>(b) * M * kCrossN;

        // B/KT is shared by every query row inside the same head/batch.
        for (int i = 0; i < M; ++i) {
            const float* __restrict__ q =
                Ab + static_cast<size_t>(i) * kQK_K;

            float* __restrict__ out =
                Cb + static_cast<size_t>(i) * kCrossN;

            qk_k64_n1500_row_v4(
                q,
                Bb,
                out
            );
        }
    }
}


// ============================================================================
// v3 dynamic QK -- M=1, K=64, N=dynamic
// ============================================================================

__attribute__((noinline))
static void whisper_qk_m1_k64_v3(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BH,
    int N) {

    const int vlmax8 =
        static_cast<int>(__riscv_vsetvlmax_e32m8());

    const int dual_cols = 2 * vlmax8;

    for (int b = 0; b < BH; ++b) {
        const float* __restrict__ q =
            A + static_cast<size_t>(b) * kQK_K;

        const float* __restrict__ kt =
            B + static_cast<size_t>(b) * kQK_K * N;

        float* __restrict__ out =
            C + static_cast<size_t>(b) * N;

        int col = 0;

        while (N - col >= dual_cols) {
            qk_dual_m8_k64(
                q,
                kt,
                out,
                N,
                col,
                vlmax8
            );

            col += dual_cols;
        }

        if (col < N) {
            qk_hierarchical_tail_k64(
                q,
                kt,
                out,
                N,
                col
            );
        }
    }
}


// ============================================================================
// PV helpers -- M=1, N=64, K=dynamic
// ============================================================================

static inline void pv_range_m8(
    const float* __restrict__ p,
    const float* __restrict__ v,
    float* __restrict__ out,
    int K,
    int N,
    int col,
    int width) {

    const size_t vl =
        __riscv_vsetvl_e32m8(static_cast<size_t>(width));

    vfloat32m8_t acc =
        __riscv_vfmv_v_f_f32m8(0.0f, vl);

    vfloat32m8_t vv;

    int k = 0;

    for (; k + 3 < K; k += 4) {
        vv = __riscv_vle32_v_f32m8(
            v + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(
            acc, p[k + 0], vv, vl);

        vv = __riscv_vle32_v_f32m8(
            v + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(
            acc, p[k + 1], vv, vl);

        vv = __riscv_vle32_v_f32m8(
            v + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(
            acc, p[k + 2], vv, vl);

        vv = __riscv_vle32_v_f32m8(
            v + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m8(
            acc, p[k + 3], vv, vl);
    }

    for (; k < K; ++k) {
        vv = __riscv_vle32_v_f32m8(
            v + static_cast<size_t>(k) * N + col, vl);

        acc = __riscv_vfmacc_vf_f32m8(
            acc, p[k], vv, vl);
    }

    __riscv_vse32_v_f32m8(
        out + col,
        acc,
        vl
    );
}


static inline void pv_range_m4(
    const float* __restrict__ p,
    const float* __restrict__ v,
    float* __restrict__ out,
    int K,
    int N,
    int col,
    int width) {

    const size_t vl =
        __riscv_vsetvl_e32m4(static_cast<size_t>(width));

    vfloat32m4_t acc =
        __riscv_vfmv_v_f_f32m4(0.0f, vl);

    vfloat32m4_t vv;

    int k = 0;

    for (; k + 3 < K; k += 4) {
        vv = __riscv_vle32_v_f32m4(
            v + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(
            acc, p[k + 0], vv, vl);

        vv = __riscv_vle32_v_f32m4(
            v + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(
            acc, p[k + 1], vv, vl);

        vv = __riscv_vle32_v_f32m4(
            v + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(
            acc, p[k + 2], vv, vl);

        vv = __riscv_vle32_v_f32m4(
            v + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m4(
            acc, p[k + 3], vv, vl);
    }

    for (; k < K; ++k) {
        vv = __riscv_vle32_v_f32m4(
            v + static_cast<size_t>(k) * N + col, vl);

        acc = __riscv_vfmacc_vf_f32m4(
            acc, p[k], vv, vl);
    }

    __riscv_vse32_v_f32m4(
        out + col,
        acc,
        vl
    );
}


static inline void pv_range_m2(
    const float* __restrict__ p,
    const float* __restrict__ v,
    float* __restrict__ out,
    int K,
    int N,
    int col,
    int width) {

    const size_t vl =
        __riscv_vsetvl_e32m2(static_cast<size_t>(width));

    vfloat32m2_t acc =
        __riscv_vfmv_v_f_f32m2(0.0f, vl);

    vfloat32m2_t vv;

    int k = 0;

    for (; k + 3 < K; k += 4) {
        vv = __riscv_vle32_v_f32m2(
            v + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(
            acc, p[k + 0], vv, vl);

        vv = __riscv_vle32_v_f32m2(
            v + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(
            acc, p[k + 1], vv, vl);

        vv = __riscv_vle32_v_f32m2(
            v + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(
            acc, p[k + 2], vv, vl);

        vv = __riscv_vle32_v_f32m2(
            v + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m2(
            acc, p[k + 3], vv, vl);
    }

    for (; k < K; ++k) {
        vv = __riscv_vle32_v_f32m2(
            v + static_cast<size_t>(k) * N + col, vl);

        acc = __riscv_vfmacc_vf_f32m2(
            acc, p[k], vv, vl);
    }

    __riscv_vse32_v_f32m2(
        out + col,
        acc,
        vl
    );
}


static inline void pv_range_m1(
    const float* __restrict__ p,
    const float* __restrict__ v,
    float* __restrict__ out,
    int K,
    int N,
    int col,
    int width) {

    const size_t vl =
        __riscv_vsetvl_e32m1(static_cast<size_t>(width));

    vfloat32m1_t acc =
        __riscv_vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t vv;

    int k = 0;

    for (; k + 3 < K; k += 4) {
        vv = __riscv_vle32_v_f32m1(
            v + static_cast<size_t>(k + 0) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(
            acc, p[k + 0], vv, vl);

        vv = __riscv_vle32_v_f32m1(
            v + static_cast<size_t>(k + 1) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(
            acc, p[k + 1], vv, vl);

        vv = __riscv_vle32_v_f32m1(
            v + static_cast<size_t>(k + 2) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(
            acc, p[k + 2], vv, vl);

        vv = __riscv_vle32_v_f32m1(
            v + static_cast<size_t>(k + 3) * N + col, vl);
        acc = __riscv_vfmacc_vf_f32m1(
            acc, p[k + 3], vv, vl);
    }

    for (; k < K; ++k) {
        vv = __riscv_vle32_v_f32m1(
            v + static_cast<size_t>(k) * N + col, vl);

        acc = __riscv_vfmacc_vf_f32m1(
            acc, p[k], vv, vl);
    }

    __riscv_vse32_v_f32m1(
        out + col,
        acc,
        vl
    );
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

    vfloat32m8_t acc0 =
        __riscv_vfmv_v_f_f32m8(0.0f, vl);

    vfloat32m8_t acc1 =
        __riscv_vfmv_v_f_f32m8(0.0f, vl);

    vfloat32m8_t vv;

    const int col1 =
        col + static_cast<int>(vl);

    int k = 0;

    for (; k + 3 < K; k += 4) {
        const float* row0 =
            v + static_cast<size_t>(k + 0) * N;

        vv = __riscv_vle32_v_f32m8(
            row0 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(
            acc0, p[k + 0], vv, vl);

        vv = __riscv_vle32_v_f32m8(
            row0 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(
            acc1, p[k + 0], vv, vl);


        const float* row1 =
            v + static_cast<size_t>(k + 1) * N;

        vv = __riscv_vle32_v_f32m8(
            row1 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(
            acc0, p[k + 1], vv, vl);

        vv = __riscv_vle32_v_f32m8(
            row1 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(
            acc1, p[k + 1], vv, vl);


        const float* row2 =
            v + static_cast<size_t>(k + 2) * N;

        vv = __riscv_vle32_v_f32m8(
            row2 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(
            acc0, p[k + 2], vv, vl);

        vv = __riscv_vle32_v_f32m8(
            row2 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(
            acc1, p[k + 2], vv, vl);


        const float* row3 =
            v + static_cast<size_t>(k + 3) * N;

        vv = __riscv_vle32_v_f32m8(
            row3 + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(
            acc0, p[k + 3], vv, vl);

        vv = __riscv_vle32_v_f32m8(
            row3 + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(
            acc1, p[k + 3], vv, vl);
    }

    for (; k < K; ++k) {
        const float* row =
            v + static_cast<size_t>(k) * N;

        vv = __riscv_vle32_v_f32m8(
            row + col, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(
            acc0, p[k], vv, vl);

        vv = __riscv_vle32_v_f32m8(
            row + col1, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(
            acc1, p[k], vv, vl);
    }

    __riscv_vse32_v_f32m8(
        out + col,
        acc0,
        vl
    );

    __riscv_vse32_v_f32m8(
        out + col1,
        acc1,
        vl
    );
}


static inline void pv_hierarchical_tail(
    const float* __restrict__ p,
    const float* __restrict__ v,
    float* __restrict__ out,
    int K,
    int N,
    int start_col) {

    int col = start_col;

    const int vlmax8 =
        static_cast<int>(__riscv_vsetvlmax_e32m8());

    const int vlmax4 =
        static_cast<int>(__riscv_vsetvlmax_e32m4());

    const int vlmax2 =
        static_cast<int>(__riscv_vsetvlmax_e32m2());

    const int vlmax1 =
        static_cast<int>(__riscv_vsetvlmax_e32m1());

    while (N - col >= vlmax8) {
        pv_range_m8(
            p, v, out, K, N, col, vlmax8);
        col += vlmax8;
    }

    if (N - col >= vlmax4) {
        pv_range_m4(
            p, v, out, K, N, col, vlmax4);
        col += vlmax4;
    }

    if (N - col >= vlmax2) {
        pv_range_m2(
            p, v, out, K, N, col, vlmax2);
        col += vlmax2;
    }

    while (col < N) {
        const int width =
            (N - col >= vlmax1)
                ? vlmax1
                : (N - col);

        pv_range_m1(
            p, v, out, K, N, col, width);

        col += width;
    }
}


__attribute__((noinline))
static void whisper_pv_m1_n64_v3(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BH,
    int K) {

    constexpr int N = 64;

    const int vlmax8 =
        static_cast<int>(__riscv_vsetvlmax_e32m8());

    const int dual_cols =
        2 * vlmax8;

    for (int b = 0; b < BH; ++b) {
        const float* __restrict__ p =
            A + static_cast<size_t>(b) * K;

        const float* __restrict__ v =
            B + static_cast<size_t>(b) * K * N;

        float* __restrict__ out =
            C + static_cast<size_t>(b) * N;

        int col = 0;

        while (N - col >= dual_cols) {
            pv_dual_m8(
                p,
                v,
                out,
                K,
                N,
                col,
                vlmax8
            );

            col += dual_cols;
        }

        if (col < N) {
            pv_hierarchical_tail(
                p,
                v,
                out,
                K,
                N,
                col
            );
        }
    }
}


// ============================================================================
// Generic RVV fallback
// ============================================================================
//
// Used for encoder Attention, unusual decoder shapes, or any BMM that does not
// match the specialized Whisper decoder QK/PV cases.
//
// K loop remains unrolled x4, but the temporary B vector is immediately
// consumed (Load -> FMA -> reuse) rather than keeping four m4 vectors alive.
// ============================================================================

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
                    __riscv_vsetvl_e32m4(
                        static_cast<size_t>(N - col)
                    );

                vfloat32m4_t acc =
                    __riscv_vfmv_v_f_f32m4(
                        0.0f,
                        vl
                    );

                vfloat32m4_t bv;

                int k = 0;

                for (; k + 3 < K; k += 4) {
                    bv = __riscv_vle32_v_f32m4(
                        Bb + static_cast<size_t>(k + 0) * N + col,
                        vl
                    );
                    acc = __riscv_vfmacc_vf_f32m4(
                        acc,
                        arow[k + 0],
                        bv,
                        vl
                    );

                    bv = __riscv_vle32_v_f32m4(
                        Bb + static_cast<size_t>(k + 1) * N + col,
                        vl
                    );
                    acc = __riscv_vfmacc_vf_f32m4(
                        acc,
                        arow[k + 1],
                        bv,
                        vl
                    );

                    bv = __riscv_vle32_v_f32m4(
                        Bb + static_cast<size_t>(k + 2) * N + col,
                        vl
                    );
                    acc = __riscv_vfmacc_vf_f32m4(
                        acc,
                        arow[k + 2],
                        bv,
                        vl
                    );

                    bv = __riscv_vle32_v_f32m4(
                        Bb + static_cast<size_t>(k + 3) * N + col,
                        vl
                    );
                    acc = __riscv_vfmacc_vf_f32m4(
                        acc,
                        arow[k + 3],
                        bv,
                        vl
                    );
                }

                for (; k < K; ++k) {
                    bv = __riscv_vle32_v_f32m4(
                        Bb + static_cast<size_t>(k) * N + col,
                        vl
                    );

                    acc = __riscv_vfmacc_vf_f32m4(
                        acc,
                        arow[k],
                        bv,
                        vl
                    );
                }

                __riscv_vse32_v_f32m4(
                    crow + col,
                    acc,
                    vl
                );

                col += static_cast<int>(vl);
            }
        }
    }
}


// Shared-B helper:
//   A [BH,M,K]
//   B [K,N]
//   C [BH,M,N]
//
// Reuse the same public dispatcher with BH=1 for each batch/head.
static inline void attention_bmm_shared_b(
    const float* A,
    const float* B,
    float* C,
    int BH,
    int M,
    int K,
    int N);

}  // namespace


// ============================================================================
// Public raw-pointer BMM entry
// ============================================================================

extern "C"
void attention_bmm_rvv_f32(
    const float* A,
    const float* B,
    float* C,
    int BH,
    int M,
    int K,
    int N) {

    if (A == nullptr ||
        B == nullptr ||
        C == nullptr ||
        BH <= 0 ||
        M <= 0 ||
        K <= 0 ||
        N <= 0) {
        return;
    }

    // ------------------------------------------------------------------------
    // Priority 1: v4 emphasized N=1500 path.
    //
    // NOTE:
    //   This is NOT:
    //       BH == 6 && M == 1 && K == 64 && N == 1500
    //
    //   It is intentionally:
    //       K == 64 && N == 1500
    //
    //   so it can still process different BH and M values.
    // ------------------------------------------------------------------------
    if (K == 64 && N == 1500) {
        whisper_qk_k64_n1500_v4(
            A,
            B,
            C,
            BH,
            M
        );
        return;
    }

    // ------------------------------------------------------------------------
    // Priority 2: v3 decoder QK -- dynamic KV length.
    // Example:
    //   [BH,1,64] x [BH,64,8/16/32/.../1499/...]
    // ------------------------------------------------------------------------
    if (M == 1 && K == 64) {
        whisper_qk_m1_k64_v3(
            A,
            B,
            C,
            BH,
            N
        );
        return;
    }

    // ------------------------------------------------------------------------
    // Priority 3: v3 decoder PV -- dynamic KV length, fixed head_dim=64.
    // Example:
    //   [BH,1,KV] x [BH,KV,64]
    // ------------------------------------------------------------------------
    if (M == 1 && N == 64) {
        whisper_pv_m1_n64_v3(
            A,
            B,
            C,
            BH,
            K
        );
        return;
    }

    // ------------------------------------------------------------------------
    // Priority 4: generic Attention BMM.
    // ------------------------------------------------------------------------
    attention_bmm_generic_rvv_m4(
        A,
        B,
        C,
        BH,
        M,
        K,
        N
    );
}


namespace {

static inline void attention_bmm_shared_b(
    const float* A,
    const float* B,
    float* C,
    int BH,
    int M,
    int K,
    int N) {

    for (int b = 0; b < BH; ++b) {
        const float* Ab =
            A + static_cast<size_t>(b) * M * K;

        float* Cb =
            C + static_cast<size_t>(b) * M * N;

        attention_bmm_rvv_f32(
            Ab,
            B,
            Cb,
            1,
            M,
            K,
            N
        );
    }
}

}  // namespace


// ============================================================================
// Optional direct N=1500 benchmark/test entry
// ============================================================================
//
// Assumes:
//   A [BH,M,64]
//   B [BH,64,1500]
//   C [BH,M,1500]
//
// Unlike the old v4 test function, BH and M are explicit parameters.
//
extern "C"
void attention_qk_n1500_v4_f32(
    const float* A,
    const float* B,
    float* C,
    int BH,
    int M) {

    if (A == nullptr ||
        B == nullptr ||
        C == nullptr ||
        BH <= 0 ||
        M <= 0) {
        return;
    }

    whisper_qk_k64_n1500_v4(
        A,
        B,
        C,
        BH,
        M
    );
}


// Backward-compatible SERIAL benchmark entry for the old fixed benchmark:
//   [6,1,64] x [6,64,1500]
extern "C"
void attention_qk_cross1500_v4_serial_f32(
    const float* A,
    const float* B,
    float* C) {

    whisper_qk_k64_n1500_v4(
        A,
        B,
        C,
        6,
        1
    );
}


// ============================================================================
// Kiwipedia DLTensor entry
// ============================================================================
//
// Supported:
//
// 1) Batched Attention:
//      A [BH,M,K]
//      B [BH,K,N]
//      C [BH,M,N]
//
// 2) Batched A with shared B:
//      A [BH,M,K]
//      B [K,N]
//      C [BH,M,N]
//
// 3) Plain matrix multiplication:
//      A [M,K]
//      B [K,N]
//      C [M,N]
//
// This makes the operator safer to integrate even if Codegen emits a slightly
// different but mathematically equivalent Attention storage form.
// ============================================================================

extern "C"
void attention_matmul(
    std::vector<const DLTensor*>& data_entry_,
    std::vector<int64_t>& shapeA,
    std::vector<int64_t>& shapeB) {

    (void)shapeA;
    (void)shapeB;

    if (data_entry_.size() < 3) {
        std::cerr
            << "[ATTENTION_RVV_V4_COMPAT] expected A, B, C\n";
        return;
    }

    const DLTensor* A_t = data_entry_[0];
    const DLTensor* B_t = data_entry_[1];
    const DLTensor* C_t = data_entry_[2];

    if (!is_fp32(A_t) ||
        !is_fp32(B_t) ||
        !is_fp32(C_t)) {
        std::cerr
            << "[ATTENTION_RVV_V4_COMPAT] FP32 only\n";
        return;
    }

    if (!is_compact_contiguous(A_t) ||
        !is_compact_contiguous(B_t) ||
        !is_compact_contiguous(C_t)) {
        std::cerr
            << "[ATTENTION_RVV_V4_COMPAT] contiguous tensors only\n";
        return;
    }

    const float* A =
        tensor_data_const(A_t);

    const float* B =
        tensor_data_const(B_t);

    float* C =
        tensor_data_mutable(C_t);


    // ------------------------------------------------------------------------
    // Case 1:
    //   A [BH,M,K]
    //   B [BH,K,N]
    //   C [BH,M,N]
    // ------------------------------------------------------------------------
    if (A_t->ndim == 3 &&
        B_t->ndim == 3 &&
        C_t->ndim == 3) {

        const int BH =
            static_cast<int>(A_t->shape[0]);

        const int M =
            static_cast<int>(A_t->shape[1]);

        const int K =
            static_cast<int>(A_t->shape[2]);

        const int B_BH =
            static_cast<int>(B_t->shape[0]);

        const int B_K =
            static_cast<int>(B_t->shape[1]);

        const int N =
            static_cast<int>(B_t->shape[2]);

        if (B_BH != BH ||
            B_K != K ||
            static_cast<int>(C_t->shape[0]) != BH ||
            static_cast<int>(C_t->shape[1]) != M ||
            static_cast<int>(C_t->shape[2]) != N) {

            std::cerr
                << "[ATTENTION_RVV_V4_COMPAT] "
                << "3D BMM shape mismatch\n";
            return;
        }

        attention_bmm_rvv_f32(
            A,
            B,
            C,
            BH,
            M,
            K,
            N
        );

        return;
    }


    // ------------------------------------------------------------------------
    // Case 2:
    //   A [BH,M,K]
    //   B [K,N]       shared B
    //   C [BH,M,N]
    // ------------------------------------------------------------------------
    if (A_t->ndim == 3 &&
        B_t->ndim == 2 &&
        C_t->ndim == 3) {

        const int BH =
            static_cast<int>(A_t->shape[0]);

        const int M =
            static_cast<int>(A_t->shape[1]);

        const int K =
            static_cast<int>(A_t->shape[2]);

        const int B_K =
            static_cast<int>(B_t->shape[0]);

        const int N =
            static_cast<int>(B_t->shape[1]);

        if (B_K != K ||
            static_cast<int>(C_t->shape[0]) != BH ||
            static_cast<int>(C_t->shape[1]) != M ||
            static_cast<int>(C_t->shape[2]) != N) {

            std::cerr
                << "[ATTENTION_RVV_V4_COMPAT] "
                << "3Dx2D shape mismatch\n";
            return;
        }

        attention_bmm_shared_b(
            A,
            B,
            C,
            BH,
            M,
            K,
            N
        );

        return;
    }


    // ------------------------------------------------------------------------
    // Case 3:
    //   A [M,K]
    //   B [K,N]
    //   C [M,N]
    // ------------------------------------------------------------------------
    if (A_t->ndim == 2 &&
        B_t->ndim == 2 &&
        C_t->ndim == 2) {

        const int M =
            static_cast<int>(A_t->shape[0]);

        const int K =
            static_cast<int>(A_t->shape[1]);

        const int B_K =
            static_cast<int>(B_t->shape[0]);

        const int N =
            static_cast<int>(B_t->shape[1]);

        if (B_K != K ||
            static_cast<int>(C_t->shape[0]) != M ||
            static_cast<int>(C_t->shape[1]) != N) {

            std::cerr
                << "[ATTENTION_RVV_V4_COMPAT] "
                << "2D shape mismatch\n";
            return;
        }

        attention_bmm_rvv_f32(
            A,
            B,
            C,
            1,
            M,
            K,
            N
        );

        return;
    }


    std::cerr
        << "[ATTENTION_RVV_V4_COMPAT] unsupported ndim: "
        << "A=" << A_t->ndim
        << " B=" << B_t->ndim
        << " C=" << C_t->ndim
        << "\n";
}
