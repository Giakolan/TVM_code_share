#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlpack/dlpack.h>
#include <riscv_vector.h>


// ==================== M=1 SPECIALIZED RVV KERNEL ====================
/**
 * //add
 * Compute C[1][N] = A[1][K] * B[K][N].
 *
 * This specialized path avoids:
 *   1. B packing
 *   2. MC/KC/NC blocking overhead
 *   3. Repeated C load/store across K tiles
 *
 * B is row-major, so B[k][col:col+vl] is contiguous and can be
 * loaded with unit-stride RVV instructions directly.
 */
static inline void matmul_m1_rvv(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    int col = 0;

    while (col < N) {
        const size_t vl =
            __riscv_vsetvl_e32m1(static_cast<size_t>(N - col));

        // C starts from zero because this function computes the full K range.
        vfloat32m1_t vacc =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            const float a_scalar = A[k];

            // B is [K][N] row-major.
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            const vfloat32m1_t bv =
                __riscv_vle32_v_f32m1(B_vec, vl);

            vacc = __riscv_vfmacc_vf_f32m1(
                vacc,
                a_scalar,
                bv,
                vl
            );
        }

        __riscv_vse32_v_f32m1(C + col, vacc, vl);

        col += static_cast<int>(vl);
    }
}

//now M1 using
__attribute__((noinline))
static void matmul_m1_rvv_m8(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    int col = 0;

    while (col < N) {
        const size_t vl =
            __riscv_vsetvl_e32m8(
                static_cast<size_t>(N - col)
            );

        vfloat32m8_t acc =
            __riscv_vfmv_v_f_f32m8(
                0.0f,
                vl
            );

        for (int k = 0; k < K; ++k) {
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            const vfloat32m8_t bv =
                __riscv_vle32_v_f32m8(
                    B_vec,
                    vl
                );

            acc =
                __riscv_vfmacc_vf_f32m8(
                    acc,
                    A[k],
                    bv,
                    vl
                );
        }

        __riscv_vse32_v_f32m8(
            C + col,
            acc,
            vl
        );

        col += static_cast<int>(vl);
    }
}




// ==================== M=2 SPECIALIZED RVV KERNEL ====================
//benchmark
static inline void matmul_m2_rvv(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    const float* A0 = A;
    const float* A1 = A + K;

    float* C0 = C;
    float* C1 = C + N;

    int col = 0;

    while (col < N) {
        size_t vl =
            __riscv_vsetvl_e32m1(
                static_cast<size_t>(N - col)
            );

        vfloat32m1_t acc0 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        vfloat32m1_t acc1 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            vfloat32m1_t bv =
                __riscv_vle32_v_f32m1(B_vec, vl);

            acc0 = __riscv_vfmacc_vf_f32m1(
                acc0,
                A0[k],
                bv,
                vl
            );

            acc1 = __riscv_vfmacc_vf_f32m1(
                acc1,
                A1[k],
                bv,
                vl
            );
        }

        __riscv_vse32_v_f32m1(C0 + col, acc0, vl);
        __riscv_vse32_v_f32m1(C1 + col, acc1, vl);

        col += static_cast<int>(vl);
    }
}



__attribute__((noinline))
static void matmul_m2_rvv_m8_range(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int main_cols
) {
    const float* A0 = A;
    const float* A1 = A + K;

    float* C0 = C;
    float* C1 = C + N;

    int col = 0;

    while (col < main_cols) {
        const size_t vl =
            __riscv_vsetvl_e32m8(
                static_cast<size_t>(main_cols - col)
            );

        vfloat32m8_t acc0 =
            __riscv_vfmv_v_f_f32m8(0.0f, vl);

        vfloat32m8_t acc1 =
            __riscv_vfmv_v_f_f32m8(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            const vfloat32m8_t bv =
                __riscv_vle32_v_f32m8(
                    B_vec,
                    vl
                );

            acc0 =
                __riscv_vfmacc_vf_f32m8(
                    acc0,
                    A0[k],
                    bv,
                    vl
                );

            acc1 =
                __riscv_vfmacc_vf_f32m8(
                    acc1,
                    A1[k],
                    bv,
                    vl
                );
        }

        __riscv_vse32_v_f32m8(
            C0 + col,
            acc0,
            vl
        );

        __riscv_vse32_v_f32m8(
            C1 + col,
            acc1,
            vl
        );

        col += static_cast<int>(vl);
    }
}





__attribute__((noinline))
static void matmul_m2_rvv_m4_range(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int start_col,
    int end_col
) {
    const float* A0 = A;
    const float* A1 = A + K;

    float* C0 = C;
    float* C1 = C + N;

    int col = start_col;

    while (col < end_col) {
        const size_t vl =
            __riscv_vsetvl_e32m4(
                static_cast<size_t>(end_col - col)
            );

        vfloat32m4_t acc0 =
            __riscv_vfmv_v_f_f32m4(0.0f, vl);

        vfloat32m4_t acc1 =
            __riscv_vfmv_v_f_f32m4(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            const vfloat32m4_t bv =
                __riscv_vle32_v_f32m4(
                    B_vec,
                    vl
                );

            acc0 =
                __riscv_vfmacc_vf_f32m4(
                    acc0,
                    A0[k],
                    bv,
                    vl
                );

            acc1 =
                __riscv_vfmacc_vf_f32m4(
                    acc1,
                    A1[k],
                    bv,
                    vl
                );
        }

        __riscv_vse32_v_f32m4(
            C0 + col,
            acc0,
            vl
        );

        __riscv_vse32_v_f32m4(
            C1 + col,
            acc1,
            vl
        );

        col += static_cast<int>(vl);
    }
}




__attribute__((noinline))
static void matmul_m2_rvv_m2_range(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int start_col,
    int end_col
) {
    const float* A0 = A;
    const float* A1 = A + K;

    float* C0 = C;
    float* C1 = C + N;

    int col = start_col;

    while (col < end_col) {
        const size_t vl =
            __riscv_vsetvl_e32m2(
                static_cast<size_t>(end_col - col)
            );

        vfloat32m2_t acc0 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);

        vfloat32m2_t acc1 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            const vfloat32m2_t bv =
                __riscv_vle32_v_f32m2(
                    B_vec,
                    vl
                );

            acc0 =
                __riscv_vfmacc_vf_f32m2(
                    acc0,
                    A0[k],
                    bv,
                    vl
                );

            acc1 =
                __riscv_vfmacc_vf_f32m2(
                    acc1,
                    A1[k],
                    bv,
                    vl
                );
        }

        __riscv_vse32_v_f32m2(
            C0 + col,
            acc0,
            vl
        );

        __riscv_vse32_v_f32m2(
            C1 + col,
            acc1,
            vl
        );

        col += static_cast<int>(vl);
    }
}



__attribute__((noinline))
static void matmul_m2_rvv_m1_range(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int start_col
) {
    const float* A0 = A;
    const float* A1 = A + K;

    float* C0 = C;
    float* C1 = C + N;

    int col = start_col;

    while (col < N) {
        const size_t vl =
            __riscv_vsetvl_e32m1(
                static_cast<size_t>(N - col)
            );

        vfloat32m1_t acc0 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        vfloat32m1_t acc1 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            const vfloat32m1_t bv =
                __riscv_vle32_v_f32m1(
                    B_vec,
                    vl
                );

            acc0 =
                __riscv_vfmacc_vf_f32m1(
                    acc0,
                    A0[k],
                    bv,
                    vl
                );

            acc1 =
                __riscv_vfmacc_vf_f32m1(
                    acc1,
                    A1[k],
                    bv,
                    vl
                );
        }

        __riscv_vse32_v_f32m1(
            C0 + col,
            acc0,
            vl
        );

        __riscv_vse32_v_f32m1(
            C1 + col,
            acc1,
            vl
        );

        col += static_cast<int>(vl);
    }
}






static void matmul_m2_rvv_hierarchical_hybrid(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    int col = 0;

    // Full 32-column blocks
    const int m8_end =
        N - (N % 32);

    if (m8_end > 0) {
        matmul_m2_rvv_m8_range(
            A,
            B,
            C,
            K,
            N,
            m8_end
        );

        col = m8_end;
    }

    // One 16-column block if possible
    if (N - col >= 16) {
        matmul_m2_rvv_m4_range(
            A,
            B,
            C,
            K,
            N,
            col,
            col + 16
        );

        col += 16;
    }

    // One 8-column block if possible
    if (N - col >= 8) {
        matmul_m2_rvv_m2_range(
            A,
            B,
            C,
            K,
            N,
            col,
            col + 8
        );

        col += 8;
    }

    // Remaining 1~7 columns
    if (col < N) {
        matmul_m2_rvv_m1_range(
            A,
            B,
            C,
            K,
            N,
            col
        );
    }
}




// ==================== M=4 SPECIALIZED RVV KERNEL ====================
static inline void matmul_m4_rvv(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    const float* A0 = A;
    const float* A1 = A + K;
    const float* A2 = A + 2 * K;
    const float* A3 = A + 3 * K;

    float* C0 = C;
    float* C1 = C + N;
    float* C2 = C + 2 * N;
    float* C3 = C + 3 * N;

    int col = 0;

    while (col < N) {
        size_t vl =
            __riscv_vsetvl_e32m1(
                static_cast<size_t>(N - col)
            );

        vfloat32m1_t acc0 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        vfloat32m1_t acc1 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        vfloat32m1_t acc2 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        vfloat32m1_t acc3 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            vfloat32m1_t bv =
                __riscv_vle32_v_f32m1(B_vec, vl);

            acc0 = __riscv_vfmacc_vf_f32m1(
                acc0,
                A0[k],
                bv,
                vl
            );

            acc1 = __riscv_vfmacc_vf_f32m1(
                acc1,
                A1[k],
                bv,
                vl
            );

            acc2 = __riscv_vfmacc_vf_f32m1(
                acc2,
                A2[k],
                bv,
                vl
            );

            acc3 = __riscv_vfmacc_vf_f32m1(
                acc3,
                A3[k],
                bv,
                vl
            );
        }

        __riscv_vse32_v_f32m1(C0 + col, acc0, vl);
        __riscv_vse32_v_f32m1(C1 + col, acc1, vl);
        __riscv_vse32_v_f32m1(C2 + col, acc2, vl);
        __riscv_vse32_v_f32m1(C3 + col, acc3, vl);

        col += static_cast<int>(vl);
    }
}


__attribute__((noinline))
static void matmul_m4_rvv_m4_body(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int main_cols
) {
    const float* A0 = A;
    const float* A1 = A + K;
    const float* A2 = A + 2 * K;
    const float* A3 = A + 3 * K;

    float* C0 = C;
    float* C1 = C + N;
    float* C2 = C + 2 * N;
    float* C3 = C + 3 * N;

    int col = 0;

    while (col < main_cols) {
        const size_t vl =
            __riscv_vsetvl_e32m4(
                static_cast<size_t>(main_cols - col)
            );

        vfloat32m4_t acc0 =
            __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc1 =
            __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc2 =
            __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc3 =
            __riscv_vfmv_v_f_f32m4(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            const vfloat32m4_t bv =
                __riscv_vle32_v_f32m4(
                    B_vec,
                    vl
                );

            acc0 =
                __riscv_vfmacc_vf_f32m4(
                    acc0,
                    A0[k],
                    bv,
                    vl
                );

            acc1 =
                __riscv_vfmacc_vf_f32m4(
                    acc1,
                    A1[k],
                    bv,
                    vl
                );

            acc2 =
                __riscv_vfmacc_vf_f32m4(
                    acc2,
                    A2[k],
                    bv,
                    vl
                );

            acc3 =
                __riscv_vfmacc_vf_f32m4(
                    acc3,
                    A3[k],
                    bv,
                    vl
                );
        }

        __riscv_vse32_v_f32m4(
            C0 + col,
            acc0,
            vl
        );

        __riscv_vse32_v_f32m4(
            C1 + col,
            acc1,
            vl
        );

        __riscv_vse32_v_f32m4(
            C2 + col,
            acc2,
            vl
        );

        __riscv_vse32_v_f32m4(
            C3 + col,
            acc3,
            vl
        );

        col += static_cast<int>(vl);
    }
}

__attribute__((noinline))
static void matmul_m4_rvv_m1_tail(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int start_col
) {
    const float* A0 = A;
    const float* A1 = A + K;
    const float* A2 = A + 2 * K;
    const float* A3 = A + 3 * K;

    float* C0 = C;
    float* C1 = C + N;
    float* C2 = C + 2 * N;
    float* C3 = C + 3 * N;

    int col = start_col;

    while (col < N) {
        const size_t vl =
            __riscv_vsetvl_e32m1(
                static_cast<size_t>(N - col)
            );

        vfloat32m1_t acc0 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc1 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc2 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc3 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            const vfloat32m1_t bv =
                __riscv_vle32_v_f32m1(
                    B_vec,
                    vl
                );

            acc0 =
                __riscv_vfmacc_vf_f32m1(
                    acc0,
                    A0[k],
                    bv,
                    vl
                );

            acc1 =
                __riscv_vfmacc_vf_f32m1(
                    acc1,
                    A1[k],
                    bv,
                    vl
                );

            acc2 =
                __riscv_vfmacc_vf_f32m1(
                    acc2,
                    A2[k],
                    bv,
                    vl
                );

            acc3 =
                __riscv_vfmacc_vf_f32m1(
                    acc3,
                    A3[k],
                    bv,
                    vl
                );
        }

        __riscv_vse32_v_f32m1(
            C0 + col,
            acc0,
            vl
        );

        __riscv_vse32_v_f32m1(
            C1 + col,
            acc1,
            vl
        );

        __riscv_vse32_v_f32m1(
            C2 + col,
            acc2,
            vl
        );

        __riscv_vse32_v_f32m1(
            C3 + col,
            acc3,
            vl
        );

        col += static_cast<int>(vl);
    }
}

static void matmul_m4_rvv_m4_hybrid(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    const int main_cols =
        N - (N % 16);

    if (main_cols > 0) {
        matmul_m4_rvv_m4_body(
            A,
            B,
            C,
            K,
            N,
            main_cols
        );
    }

    if (main_cols < N) {
        matmul_m4_rvv_m1_tail(
            A,
            B,
            C,
            K,
            N,
            main_cols
        );
    }
}

__attribute__((noinline))
static void matmul_m4_rvv_m2_range(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int start_col,
    int end_col
) {
    const float* A0 = A;
    const float* A1 = A + K;
    const float* A2 = A + static_cast<size_t>(2) * K;
    const float* A3 = A + static_cast<size_t>(3) * K;

    float* C0 = C;
    float* C1 = C + N;
    float* C2 = C + static_cast<size_t>(2) * N;
    float* C3 = C + static_cast<size_t>(3) * N;

    int col = start_col;

    while (col < end_col) {
        const size_t vl =
            __riscv_vsetvl_e32m2(
                static_cast<size_t>(end_col - col)
            );

        vfloat32m2_t acc0 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);

        vfloat32m2_t acc1 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);

        vfloat32m2_t acc2 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);

        vfloat32m2_t acc3 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            const vfloat32m2_t bv =
                __riscv_vle32_v_f32m2(
                    B_vec,
                    vl
                );

            acc0 =
                __riscv_vfmacc_vf_f32m2(
                    acc0,
                    A0[k],
                    bv,
                    vl
                );

            acc1 =
                __riscv_vfmacc_vf_f32m2(
                    acc1,
                    A1[k],
                    bv,
                    vl
                );

            acc2 =
                __riscv_vfmacc_vf_f32m2(
                    acc2,
                    A2[k],
                    bv,
                    vl
                );

            acc3 =
                __riscv_vfmacc_vf_f32m2(
                    acc3,
                    A3[k],
                    bv,
                    vl
                );
        }

        __riscv_vse32_v_f32m2(
            C0 + col,
            acc0,
            vl
        );

        __riscv_vse32_v_f32m2(
            C1 + col,
            acc1,
            vl
        );

        __riscv_vse32_v_f32m2(
            C2 + col,
            acc2,
            vl
        );

        __riscv_vse32_v_f32m2(
            C3 + col,
            acc3,
            vl
        );

        col += static_cast<int>(vl);
    }
}


static void matmul_m4_rvv_hierarchical(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    int col = 0;

    // Full 16-column blocks -> m4
    const int m4_end =
        N - (N % 16);

    if (m4_end > 0) {
        matmul_m4_rvv_m4_body(
            A,
            B,
            C,
            K,
            N,
            m4_end
        );

        col = m4_end;
    }

    // One 8-column block -> m2
    if (N - col >= 8) {
        matmul_m4_rvv_m2_range(
            A,
            B,
            C,
            K,
            N,
            col,
            col + 8
        );

        col += 8;
    }

    // Remaining 1~7 -> m1
    if (col < N) {
        matmul_m4_rvv_m1_tail(
            A,
            B,
            C,
            K,
            N,
            col
        );
    }
}


// ==================== M=8 SPECIALIZED RVV KERNEL ====================
__attribute__((noinline))
static void matmul_m8_rvv(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    const float* A0 = A;
    const float* A1 = A + K;
    const float* A2 = A + 2 * K;
    const float* A3 = A + 3 * K;
    const float* A4 = A + 4 * K;
    const float* A5 = A + 5 * K;
    const float* A6 = A + 6 * K;
    const float* A7 = A + 7 * K;

    float* C0 = C;
    float* C1 = C + N;
    float* C2 = C + 2 * N;
    float* C3 = C + 3 * N;
    float* C4 = C + 4 * N;
    float* C5 = C + 5 * N;
    float* C6 = C + 6 * N;
    float* C7 = C + 7 * N;

    int col = 0;

    while (col < N) {
        size_t vl =
            __riscv_vsetvl_e32m1(
                static_cast<size_t>(N - col)
            );

        vfloat32m1_t acc0 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        vfloat32m1_t acc1 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        vfloat32m1_t acc2 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        vfloat32m1_t acc3 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        vfloat32m1_t acc4 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        vfloat32m1_t acc5 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        vfloat32m1_t acc6 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        vfloat32m1_t acc7 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            vfloat32m1_t bv =
                __riscv_vle32_v_f32m1(B_vec, vl);

            acc0 = __riscv_vfmacc_vf_f32m1(
                acc0,
                A0[k],
                bv,
                vl
            );

            acc1 = __riscv_vfmacc_vf_f32m1(
                acc1,
                A1[k],
                bv,
                vl
            );

            acc2 = __riscv_vfmacc_vf_f32m1(
                acc2,
                A2[k],
                bv,
                vl
            );

            acc3 = __riscv_vfmacc_vf_f32m1(
                acc3,
                A3[k],
                bv,
                vl
            );
            acc4 = __riscv_vfmacc_vf_f32m1(
                acc4,
                A4[k],
                bv,
                vl
            );
            acc5 = __riscv_vfmacc_vf_f32m1(
                acc5,
                A5[k],
                bv,
                vl
            );
            acc6 = __riscv_vfmacc_vf_f32m1(
                acc6,
                A6[k],
                bv,
                vl
            );
            acc7 = __riscv_vfmacc_vf_f32m1(
                acc7,
                A7[k],
                bv,
                vl
            );
        }

        __riscv_vse32_v_f32m1(C0 + col, acc0, vl);
        __riscv_vse32_v_f32m1(C1 + col, acc1, vl);
        __riscv_vse32_v_f32m1(C2 + col, acc2, vl);
        __riscv_vse32_v_f32m1(C3 + col, acc3, vl);
        __riscv_vse32_v_f32m1(C4 + col, acc4, vl);
        __riscv_vse32_v_f32m1(C5 + col, acc5, vl);
        __riscv_vse32_v_f32m1(C6 + col, acc6, vl);
        __riscv_vse32_v_f32m1(C7 + col, acc7, vl);

        col += static_cast<int>(vl);
    }
}


__attribute__((noinline))
static void matmul_m8_rvv_m2_body(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int main_cols
) {
    const float* A0 = A;
    const float* A1 = A + static_cast<size_t>(1) * K;
    const float* A2 = A + static_cast<size_t>(2) * K;
    const float* A3 = A + static_cast<size_t>(3) * K;
    const float* A4 = A + static_cast<size_t>(4) * K;
    const float* A5 = A + static_cast<size_t>(5) * K;
    const float* A6 = A + static_cast<size_t>(6) * K;
    const float* A7 = A + static_cast<size_t>(7) * K;

    float* C0 = C;
    float* C1 = C + static_cast<size_t>(1) * N;
    float* C2 = C + static_cast<size_t>(2) * N;
    float* C3 = C + static_cast<size_t>(3) * N;
    float* C4 = C + static_cast<size_t>(4) * N;
    float* C5 = C + static_cast<size_t>(5) * N;
    float* C6 = C + static_cast<size_t>(6) * N;
    float* C7 = C + static_cast<size_t>(7) * N;

    int col = 0;

    while (col < main_cols) {
        const size_t vl =
            __riscv_vsetvl_e32m2(
                static_cast<size_t>(main_cols - col)
            );

        vfloat32m2_t acc0 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc1 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc2 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc3 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc4 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc5 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc6 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc7 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            const vfloat32m2_t bv =
                __riscv_vle32_v_f32m2(
                    B_vec,
                    vl
                );

            acc0 = __riscv_vfmacc_vf_f32m2(
                acc0, A0[k], bv, vl
            );
            acc1 = __riscv_vfmacc_vf_f32m2(
                acc1, A1[k], bv, vl
            );
            acc2 = __riscv_vfmacc_vf_f32m2(
                acc2, A2[k], bv, vl
            );
            acc3 = __riscv_vfmacc_vf_f32m2(
                acc3, A3[k], bv, vl
            );
            acc4 = __riscv_vfmacc_vf_f32m2(
                acc4, A4[k], bv, vl
            );
            acc5 = __riscv_vfmacc_vf_f32m2(
                acc5, A5[k], bv, vl
            );
            acc6 = __riscv_vfmacc_vf_f32m2(
                acc6, A6[k], bv, vl
            );
            acc7 = __riscv_vfmacc_vf_f32m2(
                acc7, A7[k], bv, vl
            );
        }

        __riscv_vse32_v_f32m2(
            C0 + col, acc0, vl
        );
        __riscv_vse32_v_f32m2(
            C1 + col, acc1, vl
        );
        __riscv_vse32_v_f32m2(
            C2 + col, acc2, vl
        );
        __riscv_vse32_v_f32m2(
            C3 + col, acc3, vl
        );
        __riscv_vse32_v_f32m2(
            C4 + col, acc4, vl
        );
        __riscv_vse32_v_f32m2(
            C5 + col, acc5, vl
        );
        __riscv_vse32_v_f32m2(
            C6 + col, acc6, vl
        );
        __riscv_vse32_v_f32m2(
            C7 + col, acc7, vl
        );

        col += static_cast<int>(vl);
    }
}


__attribute__((noinline))
static void matmul_m8_rvv_m1_tail(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int start_col
) {
    const float* A0 = A;
    const float* A1 = A + static_cast<size_t>(1) * K;
    const float* A2 = A + static_cast<size_t>(2) * K;
    const float* A3 = A + static_cast<size_t>(3) * K;
    const float* A4 = A + static_cast<size_t>(4) * K;
    const float* A5 = A + static_cast<size_t>(5) * K;
    const float* A6 = A + static_cast<size_t>(6) * K;
    const float* A7 = A + static_cast<size_t>(7) * K;

    float* C0 = C;
    float* C1 = C + static_cast<size_t>(1) * N;
    float* C2 = C + static_cast<size_t>(2) * N;
    float* C3 = C + static_cast<size_t>(3) * N;
    float* C4 = C + static_cast<size_t>(4) * N;
    float* C5 = C + static_cast<size_t>(5) * N;
    float* C6 = C + static_cast<size_t>(6) * N;
    float* C7 = C + static_cast<size_t>(7) * N;

    int col = start_col;

    while (col < N) {
        const size_t vl =
            __riscv_vsetvl_e32m1(
                static_cast<size_t>(N - col)
            );

        vfloat32m1_t acc0 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc1 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc2 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc3 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc4 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc5 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc6 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc7 =
            __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            const float* B_vec =
                B + static_cast<size_t>(k) * N + col;

            const vfloat32m1_t bv =
                __riscv_vle32_v_f32m1(B_vec, vl);

            acc0 = __riscv_vfmacc_vf_f32m1(
                acc0, A0[k], bv, vl
            );
            acc1 = __riscv_vfmacc_vf_f32m1(
                acc1, A1[k], bv, vl
            );
            acc2 = __riscv_vfmacc_vf_f32m1(
                acc2, A2[k], bv, vl
            );
            acc3 = __riscv_vfmacc_vf_f32m1(
                acc3, A3[k], bv, vl
            );
            acc4 = __riscv_vfmacc_vf_f32m1(
                acc4, A4[k], bv, vl
            );
            acc5 = __riscv_vfmacc_vf_f32m1(
                acc5, A5[k], bv, vl
            );
            acc6 = __riscv_vfmacc_vf_f32m1(
                acc6, A6[k], bv, vl
            );
            acc7 = __riscv_vfmacc_vf_f32m1(
                acc7, A7[k], bv, vl
            );
        }

        __riscv_vse32_v_f32m1(C0 + col, acc0, vl);
        __riscv_vse32_v_f32m1(C1 + col, acc1, vl);
        __riscv_vse32_v_f32m1(C2 + col, acc2, vl);
        __riscv_vse32_v_f32m1(C3 + col, acc3, vl);
        __riscv_vse32_v_f32m1(C4 + col, acc4, vl);
        __riscv_vse32_v_f32m1(C5 + col, acc5, vl);
        __riscv_vse32_v_f32m1(C6 + col, acc6, vl);
        __riscv_vse32_v_f32m1(C7 + col, acc7, vl);

        col += static_cast<int>(vl);
    }
}


static inline void matmul_m8_rvv_m2_hybrid(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    const int main_cols =
        N - (N % 8);

    if (main_cols > 0) {
        matmul_m8_rvv_m2_body(
            A,
            B,
            C,
            K,
            N,
            main_cols
        );
    }

    if (main_cols < N) {
        matmul_m8_rvv_m1_tail(
            A,
            B,
            C,
            K,
            N,
            main_cols
        );
    }
}


static inline void matmul_rows_by_8_rvv(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = 0;


    while (row + 8 <= M) {
        matmul_m8_rvv_m2_hybrid(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );
        row += 8;
    }

    const int remain = M - row;

    if (remain & 4) {
        matmul_m4_rvv_m4_hybrid(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );
        row += 4;
    }

    if (remain & 2) {
        matmul_m2_rvv_hierarchical_hybrid(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );
        row += 2;
    }

    if (remain & 1) {
        matmul_m1_rvv_m8(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );
    }
}


static void do_block_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    matmul_rows_by_8_rvv(
        A,
        B,
        C,
        M,
        K,
        N
    );
    return;

}

// ==================== 4. BATCH PROCESSING ====================

// Batch x Batch: Each batch index has its own A, B, C
void matmul_bxb(
    std::vector<const DLTensor*>& data_entry_,
    int n, int m, int o, int batch
) {
    const float* A = static_cast<const float*>(data_entry_[0]->data);
    const float* B = static_cast<const float*>(data_entry_[1]->data);
    float* C = static_cast<float*>(data_entry_[2]->data);

    for (int b = 0; b < batch; ++b) {
        const float* A_batch = A + (size_t)b * (size_t)n * (size_t)m;
        const float* B_batch = B + (size_t)b * (size_t)m * (size_t)o;
        float* C_batch = C + (size_t)b * (size_t)n * (size_t)o;
        
        do_block_matmul(A_batch, B_batch, C_batch, n, m, o);
    }
}


void matmul_bxs(
    std::vector<const DLTensor*>& data_entry_,
    int n,
    int m,
    int o,
    int batch
) {
    const float* A =
        static_cast<const float*>(data_entry_[0]->data);

    const float* B =
        static_cast<const float*>(data_entry_[1]->data);

    float* C =
        static_cast<float*>(data_entry_[2]->data);

    const int flattened_M = batch * n;

    do_block_matmul(
        A,
        B,
        C,
        flattened_M,
        m,
        o
    );
}

// ==================== 5. MAIN ENTRY POINT ====================
static int64_t NumElements(const DLTensor* t) {
    int64_t total = 1;
    for (int i = 0; i < t->ndim; ++i) {
        total *= t->shape[i];
    }
    return total;
}

extern "C"
void matmul(
    std::vector<const DLTensor*>& data_entry_,
    std::vector<int64_t>& shapeA,
    std::vector<int64_t>& shapeB
) {

    const DLTensor* A = data_entry_[0];
    const DLTensor* B = data_entry_[1];
    const DLTensor* C = data_entry_[2];

    auto check_output_size = [&](int64_t expected) -> bool {
        int64_t actual = NumElements(C);
        if (actual != expected) {
            std::cout << "[ERROR][RVV matmul] output size mismatch. expected="
                      << expected << " actual=" << actual << std::endl;

            std::cout << "[ERROR][RVV matmul] A ndim=" << A->ndim
                      << " B ndim=" << B->ndim
                      << " C ndim=" << C->ndim << std::endl;

            return false;
        }
        return true;
    };

    if (A->ndim == 3 && B->ndim == 2) {
        // A: [batch, n, m]
        // B: [m, o]
        // C: [batch, n, o]
        int batch = static_cast<int>(A->shape[0]);
        int n     = static_cast<int>(A->shape[1]);
        int m     = static_cast<int>(A->shape[2]);
        int o     = static_cast<int>(B->shape[1]);

        int64_t expected = 1LL * batch * n * o;
        if (!check_output_size(expected)) return;

        matmul_bxs(data_entry_, n, m, o, batch);

    } else if (A->ndim == 3 && B->ndim == 3) {
        // A: [batch, n, m]
        // B: [batch, m, o]
        // C: [batch, n, o]
        int batch = static_cast<int>(A->shape[0]);
        int n     = static_cast<int>(A->shape[1]);
        int m     = static_cast<int>(A->shape[2]);
        int o     = static_cast<int>(B->shape[2]);

        int64_t expected = 1LL * batch * n * o;
        if (!check_output_size(expected)) return;

        matmul_bxb(data_entry_, n, m, o, batch);

    } else if (A->ndim == 2 && B->ndim == 2) {
        // A: [n, m]
        // B: [m, o]
        // C: [n, o]
        int n = static_cast<int>(A->shape[0]);
        int m = static_cast<int>(A->shape[1]);
        int o = static_cast<int>(B->shape[1]);

        int64_t expected = 1LL * n * o;
        if (!check_output_size(expected)) return;

        const float* a = static_cast<const float*>(A->data);
        const float* b = static_cast<const float*>(B->data);
        float* c = static_cast<float*>(C->data);

        do_block_matmul(a, b, c, n, m, o);

    } else {
        std::cout << "[ERROR][RVV matmul] unsupported ndim. A->ndim="
                  << A->ndim << " B->ndim=" << B->ndim
                  << " C->ndim=" << C->ndim << std::endl;
        return;
    }
}

/*

extern "C" void matmul_m2_m1_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m2_rvv(A, B, C, K, N);
}

extern "C" void matmul_m2_m2_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m2_rvv_m2(A, B, C, K, N);
}

extern "C" void matmul_m2_m4_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m2_rvv_m4(A, B, C, K, N);
}

extern "C" void matmul_m2_m8_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m2_rvv_m8(A, B, C, K, N);
}


extern "C" void matmul_m2_hybrid_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m2_rvv_m8_hybrid(
        A,
        B,
        C,
        K,
        N
    );
}




extern "C" void matmul_m2_hierarchical_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m2_rvv_hierarchical_hybrid(
        A,
        B,
        C,
        K,
        N
    );
}



extern "C" void matmul_m1_m1_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m1_rvv(
        A,
        B,
        C,
        K,
        N
    );
}

extern "C" void matmul_m1_m2_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m1_rvv_m2(
        A,
        B,
        C,
        K,
        N
    );
}

extern "C" void matmul_m1_m4_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m1_rvv_m4(
        A,
        B,
        C,
        K,
        N
    );
}

extern "C" void matmul_m1_m8_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m1_rvv_m8(
        A,
        B,
        C,
        K,
        N
    );
}


extern "C" void matmul_m4_hybrid_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m4_rvv_m4_hybrid(
        A,
        B,
        C,
        K,
        N
    );
}

extern "C" void matmul_m4_hierarchical_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m4_rvv_hierarchical(
        A,
        B,
        C,
        K,
        N
    );
}

*/

extern "C" void matmul_dispatch_baseline_test(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = 0;

    while (row + 8 <= M) {
        matmul_m8_rvv(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );

        row += 8;
    }

    if (row + 4 <= M) {
        matmul_m4_rvv(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );

        row += 4;
    }

    if (row + 2 <= M) {
        matmul_m2_rvv(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );

        row += 2;
    }

    if (row < M) {
        matmul_m1_rvv(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );
    }
}

extern "C" void matmul_dispatch_optimized_test(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = 0;

    while (row + 8 <= M) {
        matmul_m8_rvv_m2_hybrid(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );

        row += 8;
    }

    if (row + 4 <= M) {
        matmul_m4_rvv_m4_hybrid(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );

        row += 4;
    }

    if (row + 2 <= M) {
        matmul_m2_rvv_hierarchical_hybrid(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );

        row += 2;
    }

    if (row < M) {
        matmul_m1_rvv_m8(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );
    }
}


extern "C" void matmul_m8_original_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m8_rvv(
        A,
        B,
        C,
        K,
        N
    );
}

extern "C" void matmul_m8_hybrid_test(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    matmul_m8_rvv_m2_hybrid(
        A,
        B,
        C,
        K,
        N
    );
}