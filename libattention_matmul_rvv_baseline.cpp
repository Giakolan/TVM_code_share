#include <cstddef>
#include <cstring>
#include <riscv_vector.h>

// Original RVV MatMul baseline adapted to the standalone attention BMM interface.
// Algorithm preserved from the initial implementation:
//   - 3-level blocking (MC/NC/KC)
//   - pack B tile into contiguous [kc][nc]
//   - vectorize N dimension with RVV e32m1
//   - unit-stride vle32 + vfmacc_vf + vse32

#ifndef MC
#define MC 64
#endif
#ifndef NC
#define NC 128
#endif
#ifndef KC
#define KC 128
#endif

static inline void pack_B_tile(
    const float* B,
    int N,
    int pc,
    int kc,
    int jc,
    int nc,
    float* Bp) {
  for (int k = 0; k < kc; ++k) {
    const float* src =
        B + static_cast<size_t>(pc + k) * static_cast<size_t>(N)
          + static_cast<size_t>(jc);
    float* dst = Bp + static_cast<size_t>(k) * static_cast<size_t>(nc);
    std::memcpy(dst, src, static_cast<size_t>(nc) * sizeof(float));
  }
}

static inline void microkernel_rvv_unit_stride(
    const float* Ablk,
    int lda,
    const float* Bp,
    float* Cblk,
    int ldc,
    int mc,
    int kc,
    int nc) {
  for (int i = 0; i < mc; ++i) {
    const float* Arow =
        Ablk + static_cast<size_t>(i) * static_cast<size_t>(lda);
    float* Crow =
        Cblk + static_cast<size_t>(i) * static_cast<size_t>(ldc);

    int col = 0;
    while (col < nc) {
      const size_t vl =
          __riscv_vsetvl_e32m1(static_cast<size_t>(nc - col));

      vfloat32m1_t acc = __riscv_vle32_v_f32m1(Crow + col, vl);

      for (int k = 0; k < kc; ++k) {
        const float a = Arow[k];
        const float* Bvec =
            Bp + static_cast<size_t>(k) * static_cast<size_t>(nc)
               + static_cast<size_t>(col);
        const vfloat32m1_t bv = __riscv_vle32_v_f32m1(Bvec, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a, bv, vl);
      }

      __riscv_vse32_v_f32m1(Crow + col, acc, vl);
      col += static_cast<int>(vl);
    }
  }
}

static void do_block_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N) {
  static float Bpack[KC * NC] __attribute__((aligned(64)));

  for (int jc = 0; jc < N; jc += NC) {
    const int nc = (jc + NC <= N) ? NC : (N - jc);

    for (int ic = 0; ic < M; ic += MC) {
      const int mc = (ic + MC <= M) ? MC : (M - ic);

      for (int i = 0; i < mc; ++i) {
        float* Crow =
            C + static_cast<size_t>(ic + i) * static_cast<size_t>(N)
              + static_cast<size_t>(jc);
        std::memset(Crow, 0, static_cast<size_t>(nc) * sizeof(float));
      }

      for (int pc = 0; pc < K; pc += KC) {
        const int kc = (pc + KC <= K) ? KC : (K - pc);

        pack_B_tile(B, N, pc, kc, jc, nc, Bpack);

        const float* Atile =
            A + static_cast<size_t>(ic) * static_cast<size_t>(K)
              + static_cast<size_t>(pc);
        float* Ctile =
            C + static_cast<size_t>(ic) * static_cast<size_t>(N)
              + static_cast<size_t>(jc);

        microkernel_rvv_unit_stride(
            Atile, K, Bpack, Ctile, N, mc, kc, nc);
      }
    }
  }
}

extern "C" void attention_bmm_rvv_baseline_f32(
    const float* A,
    const float* B,
    float* C,
    int batch,
    int M,
    int K,
    int N) {
  for (int b = 0; b < batch; ++b) {
    const float* Ab =
        A + static_cast<size_t>(b) * static_cast<size_t>(M) * K;
    const float* Bb =
        B + static_cast<size_t>(b) * static_cast<size_t>(K) * N;
    float* Cb =
        C + static_cast<size_t>(b) * static_cast<size_t>(M) * N;

    do_block_matmul(Ab, Bb, Cb, M, K, N);
  }
}
