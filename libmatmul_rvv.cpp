#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlpack/dlpack.h>
#include <riscv_vector.h>
#include <chrono>
#include <map>
#include <tuple>
#include <thread>
#include <set>
#include <mutex>


static std::set<std::tuple<int,int,int>> seen_shapes;
static std::mutex seen_shapes_mutex;



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

// ==================== M=10 SPECIALIZED RVV KERNEL ====================
__attribute__((noinline))
static void matmul_m10_shared_b_rvv(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    const float* A0 = A + static_cast<size_t>(0) * K;
    const float* A1 = A + static_cast<size_t>(1) * K;
    const float* A2 = A + static_cast<size_t>(2) * K;
    const float* A3 = A + static_cast<size_t>(3) * K;
    const float* A4 = A + static_cast<size_t>(4) * K;
    const float* A5 = A + static_cast<size_t>(5) * K;
    const float* A6 = A + static_cast<size_t>(6) * K;
    const float* A7 = A + static_cast<size_t>(7) * K;
    const float* A8 = A + static_cast<size_t>(8) * K;
    const float* A9 = A + static_cast<size_t>(9) * K;

    float* C0 = C + static_cast<size_t>(0) * N;
    float* C1 = C + static_cast<size_t>(1) * N;
    float* C2 = C + static_cast<size_t>(2) * N;
    float* C3 = C + static_cast<size_t>(3) * N;
    float* C4 = C + static_cast<size_t>(4) * N;
    float* C5 = C + static_cast<size_t>(5) * N;
    float* C6 = C + static_cast<size_t>(6) * N;
    float* C7 = C + static_cast<size_t>(7) * N;
    float* C8 = C + static_cast<size_t>(8) * N;
    float* C9 = C + static_cast<size_t>(9) * N;

    int col = 0;

    while (col < N) {
        const size_t vl =
            __riscv_vsetvl_e32m2(
                static_cast<size_t>(N - col)
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
        vfloat32m2_t acc8 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc9 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);

        const float* Bp =
            B + static_cast<size_t>(col);

        for (int k = 0; k < K; ++k) {
            const vfloat32m2_t bv =
                __riscv_vle32_v_f32m2(
                    Bp,
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
            acc8 = __riscv_vfmacc_vf_f32m2(
                acc8, A8[k], bv, vl
            );
            acc9 = __riscv_vfmacc_vf_f32m2(
                acc9, A9[k], bv, vl
            );

            Bp += N;
        }

        __riscv_vse32_v_f32m2(C0 + col, acc0, vl);
        __riscv_vse32_v_f32m2(C1 + col, acc1, vl);
        __riscv_vse32_v_f32m2(C2 + col, acc2, vl);
        __riscv_vse32_v_f32m2(C3 + col, acc3, vl);
        __riscv_vse32_v_f32m2(C4 + col, acc4, vl);
        __riscv_vse32_v_f32m2(C5 + col, acc5, vl);
        __riscv_vse32_v_f32m2(C6 + col, acc6, vl);
        __riscv_vse32_v_f32m2(C7 + col, acc7, vl);
        __riscv_vse32_v_f32m2(C8 + col, acc8, vl);
        __riscv_vse32_v_f32m2(C9 + col, acc9, vl);

        col += static_cast<int>(vl);
    }
}

static inline void matmul_rows_by_10_rvv(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = 0;

    while (row + 10 <= M) {
        matmul_m10_shared_b_rvv(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );

        row += 10;
    }

    // fallback
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


__attribute__((noinline))
static void matmul_m10_shared_b_unroll4_rvv(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    const float* A0 = A + static_cast<size_t>(0) * K;
    const float* A1 = A + static_cast<size_t>(1) * K;
    const float* A2 = A + static_cast<size_t>(2) * K;
    const float* A3 = A + static_cast<size_t>(3) * K;
    const float* A4 = A + static_cast<size_t>(4) * K;
    const float* A5 = A + static_cast<size_t>(5) * K;
    const float* A6 = A + static_cast<size_t>(6) * K;
    const float* A7 = A + static_cast<size_t>(7) * K;
    const float* A8 = A + static_cast<size_t>(8) * K;
    const float* A9 = A + static_cast<size_t>(9) * K;

    float* C0 = C + static_cast<size_t>(0) * N;
    float* C1 = C + static_cast<size_t>(1) * N;
    float* C2 = C + static_cast<size_t>(2) * N;
    float* C3 = C + static_cast<size_t>(3) * N;
    float* C4 = C + static_cast<size_t>(4) * N;
    float* C5 = C + static_cast<size_t>(5) * N;
    float* C6 = C + static_cast<size_t>(6) * N;
    float* C7 = C + static_cast<size_t>(7) * N;
    float* C8 = C + static_cast<size_t>(8) * N;
    float* C9 = C + static_cast<size_t>(9) * N;

    int col = 0;

    while (col < N) {
        size_t vl =
            __riscv_vsetvl_e32m2(
                static_cast<size_t>(N - col)
            );

        vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc4 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc5 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc6 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc7 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc8 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc9 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        const float* Bp = B + col;
        int k = 0;

#define FMA10(KIDX, BPTR)                                                \
        do {                                                             \
            vfloat32m2_t bv =                                           \
                __riscv_vle32_v_f32m2((BPTR), vl);                      \
            acc0 = __riscv_vfmacc_vf_f32m2(acc0, A0[(KIDX)], bv, vl);   \
            acc1 = __riscv_vfmacc_vf_f32m2(acc1, A1[(KIDX)], bv, vl);   \
            acc2 = __riscv_vfmacc_vf_f32m2(acc2, A2[(KIDX)], bv, vl);   \
            acc3 = __riscv_vfmacc_vf_f32m2(acc3, A3[(KIDX)], bv, vl);   \
            acc4 = __riscv_vfmacc_vf_f32m2(acc4, A4[(KIDX)], bv, vl);   \
            acc5 = __riscv_vfmacc_vf_f32m2(acc5, A5[(KIDX)], bv, vl);   \
            acc6 = __riscv_vfmacc_vf_f32m2(acc6, A6[(KIDX)], bv, vl);   \
            acc7 = __riscv_vfmacc_vf_f32m2(acc7, A7[(KIDX)], bv, vl);   \
            acc8 = __riscv_vfmacc_vf_f32m2(acc8, A8[(KIDX)], bv, vl);   \
            acc9 = __riscv_vfmacc_vf_f32m2(acc9, A9[(KIDX)], bv, vl);   \
        } while (0)

        for (; k + 3 < K; k += 4) {
            FMA10(k,     Bp);
            FMA10(k + 1, Bp + static_cast<size_t>(N));
            FMA10(k + 2, Bp + static_cast<size_t>(2) * N);
            FMA10(k + 3, Bp + static_cast<size_t>(3) * N);

            Bp += static_cast<size_t>(4) * N;
        }

        for (; k < K; ++k) {
            FMA10(k, Bp);
            Bp += static_cast<size_t>(N);
        }

#undef FMA10

        __riscv_vse32_v_f32m2(C0 + col, acc0, vl);
        __riscv_vse32_v_f32m2(C1 + col, acc1, vl);
        __riscv_vse32_v_f32m2(C2 + col, acc2, vl);
        __riscv_vse32_v_f32m2(C3 + col, acc3, vl);
        __riscv_vse32_v_f32m2(C4 + col, acc4, vl);
        __riscv_vse32_v_f32m2(C5 + col, acc5, vl);
        __riscv_vse32_v_f32m2(C6 + col, acc6, vl);
        __riscv_vse32_v_f32m2(C7 + col, acc7, vl);
        __riscv_vse32_v_f32m2(C8 + col, acc8, vl);
        __riscv_vse32_v_f32m2(C9 + col, acc9, vl);

        col += static_cast<int>(vl);
    }
}

static inline void matmul_rows_by_10_unroll4_rvv(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = 0;

    while (row + 10 <= M) {
        matmul_m10_shared_b_unroll4_rvv(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );
        row += 10;
    }

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


// ==================== M=12 SPECIALIZED RVV KERNEL ====================
__attribute__((noinline))
static void matmul_m12_shared_b_unroll2_rvv(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    const float* A0  = A + static_cast<size_t>(0)  * K;
    const float* A1  = A + static_cast<size_t>(1)  * K;
    const float* A2  = A + static_cast<size_t>(2)  * K;
    const float* A3  = A + static_cast<size_t>(3)  * K;
    const float* A4  = A + static_cast<size_t>(4)  * K;
    const float* A5  = A + static_cast<size_t>(5)  * K;
    const float* A6  = A + static_cast<size_t>(6)  * K;
    const float* A7  = A + static_cast<size_t>(7)  * K;
    const float* A8  = A + static_cast<size_t>(8)  * K;
    const float* A9  = A + static_cast<size_t>(9)  * K;
    const float* A10 = A + static_cast<size_t>(10) * K;
    const float* A11 = A + static_cast<size_t>(11) * K;

    float* C0  = C + static_cast<size_t>(0)  * N;
    float* C1  = C + static_cast<size_t>(1)  * N;
    float* C2  = C + static_cast<size_t>(2)  * N;
    float* C3  = C + static_cast<size_t>(3)  * N;
    float* C4  = C + static_cast<size_t>(4)  * N;
    float* C5  = C + static_cast<size_t>(5)  * N;
    float* C6  = C + static_cast<size_t>(6)  * N;
    float* C7  = C + static_cast<size_t>(7)  * N;
    float* C8  = C + static_cast<size_t>(8)  * N;
    float* C9  = C + static_cast<size_t>(9)  * N;
    float* C10 = C + static_cast<size_t>(10) * N;
    float* C11 = C + static_cast<size_t>(11) * N;

    int col = 0;

    while (col < N) {
        const size_t vl =
            __riscv_vsetvl_e32m2(
                static_cast<size_t>(N - col)
            );

        vfloat32m2_t acc0  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc1  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc2  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc3  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc4  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc5  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc6  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc7  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc8  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc9  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc10 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc11 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        const float* Bp =
            B + static_cast<size_t>(col);

        int k = 0;

        for (; k + 1 < K; k += 2) {
            const vfloat32m2_t bv0 =
                __riscv_vle32_v_f32m2(
                    Bp,
                    vl
                );

            const vfloat32m2_t bv1 =
                __riscv_vle32_v_f32m2(
                    Bp + N,
                    vl
                );

            acc0  = __riscv_vfmacc_vf_f32m2(acc0,  A0[k],  bv0, vl);
            acc1  = __riscv_vfmacc_vf_f32m2(acc1,  A1[k],  bv0, vl);
            acc2  = __riscv_vfmacc_vf_f32m2(acc2,  A2[k],  bv0, vl);
            acc3  = __riscv_vfmacc_vf_f32m2(acc3,  A3[k],  bv0, vl);
            acc4  = __riscv_vfmacc_vf_f32m2(acc4,  A4[k],  bv0, vl);
            acc5  = __riscv_vfmacc_vf_f32m2(acc5,  A5[k],  bv0, vl);
            acc6  = __riscv_vfmacc_vf_f32m2(acc6,  A6[k],  bv0, vl);
            acc7  = __riscv_vfmacc_vf_f32m2(acc7,  A7[k],  bv0, vl);
            acc8  = __riscv_vfmacc_vf_f32m2(acc8,  A8[k],  bv0, vl);
            acc9  = __riscv_vfmacc_vf_f32m2(acc9,  A9[k],  bv0, vl);
            acc10 = __riscv_vfmacc_vf_f32m2(acc10, A10[k], bv0, vl);
            acc11 = __riscv_vfmacc_vf_f32m2(acc11, A11[k], bv0, vl);

            acc0  = __riscv_vfmacc_vf_f32m2(acc0,  A0[k+1],  bv1, vl);
            acc1  = __riscv_vfmacc_vf_f32m2(acc1,  A1[k+1],  bv1, vl);
            acc2  = __riscv_vfmacc_vf_f32m2(acc2,  A2[k+1],  bv1, vl);
            acc3  = __riscv_vfmacc_vf_f32m2(acc3,  A3[k+1],  bv1, vl);
            acc4  = __riscv_vfmacc_vf_f32m2(acc4,  A4[k+1],  bv1, vl);
            acc5  = __riscv_vfmacc_vf_f32m2(acc5,  A5[k+1],  bv1, vl);
            acc6  = __riscv_vfmacc_vf_f32m2(acc6,  A6[k+1],  bv1, vl);
            acc7  = __riscv_vfmacc_vf_f32m2(acc7,  A7[k+1],  bv1, vl);
            acc8  = __riscv_vfmacc_vf_f32m2(acc8,  A8[k+1],  bv1, vl);
            acc9  = __riscv_vfmacc_vf_f32m2(acc9,  A9[k+1],  bv1, vl);
            acc10 = __riscv_vfmacc_vf_f32m2(acc10, A10[k+1], bv1, vl);
            acc11 = __riscv_vfmacc_vf_f32m2(acc11, A11[k+1], bv1, vl);

            Bp += static_cast<size_t>(2) * N;
        }

        // Generic odd-K tail.
        if (k < K) {
            const vfloat32m2_t bv =
                __riscv_vle32_v_f32m2(
                    Bp,
                    vl
                );

            acc0  = __riscv_vfmacc_vf_f32m2(acc0,  A0[k],  bv, vl);
            acc1  = __riscv_vfmacc_vf_f32m2(acc1,  A1[k],  bv, vl);
            acc2  = __riscv_vfmacc_vf_f32m2(acc2,  A2[k],  bv, vl);
            acc3  = __riscv_vfmacc_vf_f32m2(acc3,  A3[k],  bv, vl);
            acc4  = __riscv_vfmacc_vf_f32m2(acc4,  A4[k],  bv, vl);
            acc5  = __riscv_vfmacc_vf_f32m2(acc5,  A5[k],  bv, vl);
            acc6  = __riscv_vfmacc_vf_f32m2(acc6,  A6[k],  bv, vl);
            acc7  = __riscv_vfmacc_vf_f32m2(acc7,  A7[k],  bv, vl);
            acc8  = __riscv_vfmacc_vf_f32m2(acc8,  A8[k],  bv, vl);
            acc9  = __riscv_vfmacc_vf_f32m2(acc9,  A9[k],  bv, vl);
            acc10 = __riscv_vfmacc_vf_f32m2(acc10, A10[k], bv, vl);
            acc11 = __riscv_vfmacc_vf_f32m2(acc11, A11[k], bv, vl);
        }

        __riscv_vse32_v_f32m2(C0  + col, acc0,  vl);
        __riscv_vse32_v_f32m2(C1  + col, acc1,  vl);
        __riscv_vse32_v_f32m2(C2  + col, acc2,  vl);
        __riscv_vse32_v_f32m2(C3  + col, acc3,  vl);
        __riscv_vse32_v_f32m2(C4  + col, acc4,  vl);
        __riscv_vse32_v_f32m2(C5  + col, acc5,  vl);
        __riscv_vse32_v_f32m2(C6  + col, acc6,  vl);
        __riscv_vse32_v_f32m2(C7  + col, acc7,  vl);
        __riscv_vse32_v_f32m2(C8  + col, acc8,  vl);
        __riscv_vse32_v_f32m2(C9  + col, acc9,  vl);
        __riscv_vse32_v_f32m2(C10 + col, acc10, vl);
        __riscv_vse32_v_f32m2(C11 + col, acc11, vl);

        col += static_cast<int>(vl);
    }
}



static inline void matmul_rows_by_12_unroll2_rvv(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = 0;

    while (row + 12 <= M) {
        matmul_m12_shared_b_unroll2_rvv(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );

        row += 12;
    }

    // Generic fallback.
    while (row + 10 <= M) {
        matmul_m10_shared_b_rvv(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );
        row += 10;
    }

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



// ==================== M=14 SPECIALIZED RVV KERNEL ====================
__attribute__((noinline))
static void matmul_m14_shared_b_unroll2_rvv(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    const float* A0  = A + static_cast<size_t>(0)  * K;
    const float* A1  = A + static_cast<size_t>(1)  * K;
    const float* A2  = A + static_cast<size_t>(2)  * K;
    const float* A3  = A + static_cast<size_t>(3)  * K;
    const float* A4  = A + static_cast<size_t>(4)  * K;
    const float* A5  = A + static_cast<size_t>(5)  * K;
    const float* A6  = A + static_cast<size_t>(6)  * K;
    const float* A7  = A + static_cast<size_t>(7)  * K;
    const float* A8  = A + static_cast<size_t>(8)  * K;
    const float* A9  = A + static_cast<size_t>(9)  * K;
    const float* A10 = A + static_cast<size_t>(10) * K;
    const float* A11 = A + static_cast<size_t>(11) * K;
    const float* A12 = A + static_cast<size_t>(12) * K;
    const float* A13 = A + static_cast<size_t>(13) * K;

    float* C0  = C + static_cast<size_t>(0)  * N;
    float* C1  = C + static_cast<size_t>(1)  * N;
    float* C2  = C + static_cast<size_t>(2)  * N;
    float* C3  = C + static_cast<size_t>(3)  * N;
    float* C4  = C + static_cast<size_t>(4)  * N;
    float* C5  = C + static_cast<size_t>(5)  * N;
    float* C6  = C + static_cast<size_t>(6)  * N;
    float* C7  = C + static_cast<size_t>(7)  * N;
    float* C8  = C + static_cast<size_t>(8)  * N;
    float* C9  = C + static_cast<size_t>(9)  * N;
    float* C10 = C + static_cast<size_t>(10) * N;
    float* C11 = C + static_cast<size_t>(11) * N;
    float* C12 = C + static_cast<size_t>(12) * N;
    float* C13 = C + static_cast<size_t>(13) * N;

    int col = 0;

    while (col < N) {
        const size_t vl =
            __riscv_vsetvl_e32m2(
                static_cast<size_t>(N - col)
            );

        vfloat32m2_t acc0  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc1  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc2  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc3  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc4  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc5  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc6  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc7  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc8  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc9  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc10 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc11 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc12 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc13 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        const float* Bp =
            B + static_cast<size_t>(col);

        int k = 0;

        for (; k + 1 < K; k += 2) {
            // k
            {
                const vfloat32m2_t bv =
                    __riscv_vle32_v_f32m2(Bp, vl);

                acc0  = __riscv_vfmacc_vf_f32m2(acc0,  A0[k],  bv, vl);
                acc1  = __riscv_vfmacc_vf_f32m2(acc1,  A1[k],  bv, vl);
                acc2  = __riscv_vfmacc_vf_f32m2(acc2,  A2[k],  bv, vl);
                acc3  = __riscv_vfmacc_vf_f32m2(acc3,  A3[k],  bv, vl);
                acc4  = __riscv_vfmacc_vf_f32m2(acc4,  A4[k],  bv, vl);
                acc5  = __riscv_vfmacc_vf_f32m2(acc5,  A5[k],  bv, vl);
                acc6  = __riscv_vfmacc_vf_f32m2(acc6,  A6[k],  bv, vl);
                acc7  = __riscv_vfmacc_vf_f32m2(acc7,  A7[k],  bv, vl);
                acc8  = __riscv_vfmacc_vf_f32m2(acc8,  A8[k],  bv, vl);
                acc9  = __riscv_vfmacc_vf_f32m2(acc9,  A9[k],  bv, vl);
                acc10 = __riscv_vfmacc_vf_f32m2(acc10, A10[k], bv, vl);
                acc11 = __riscv_vfmacc_vf_f32m2(acc11, A11[k], bv, vl);
                acc12 = __riscv_vfmacc_vf_f32m2(acc12, A12[k], bv, vl);
                acc13 = __riscv_vfmacc_vf_f32m2(acc13, A13[k], bv, vl);
            }

            // k + 1
            {
                const vfloat32m2_t bv =
                    __riscv_vle32_v_f32m2(Bp + N, vl);

                acc0  = __riscv_vfmacc_vf_f32m2(acc0,  A0[k+1],  bv, vl);
                acc1  = __riscv_vfmacc_vf_f32m2(acc1,  A1[k+1],  bv, vl);
                acc2  = __riscv_vfmacc_vf_f32m2(acc2,  A2[k+1],  bv, vl);
                acc3  = __riscv_vfmacc_vf_f32m2(acc3,  A3[k+1],  bv, vl);
                acc4  = __riscv_vfmacc_vf_f32m2(acc4,  A4[k+1],  bv, vl);
                acc5  = __riscv_vfmacc_vf_f32m2(acc5,  A5[k+1],  bv, vl);
                acc6  = __riscv_vfmacc_vf_f32m2(acc6,  A6[k+1],  bv, vl);
                acc7  = __riscv_vfmacc_vf_f32m2(acc7,  A7[k+1],  bv, vl);
                acc8  = __riscv_vfmacc_vf_f32m2(acc8,  A8[k+1],  bv, vl);
                acc9  = __riscv_vfmacc_vf_f32m2(acc9,  A9[k+1],  bv, vl);
                acc10 = __riscv_vfmacc_vf_f32m2(acc10, A10[k+1], bv, vl);
                acc11 = __riscv_vfmacc_vf_f32m2(acc11, A11[k+1], bv, vl);
                acc12 = __riscv_vfmacc_vf_f32m2(acc12, A12[k+1], bv, vl);
                acc13 = __riscv_vfmacc_vf_f32m2(acc13, A13[k+1], bv, vl);
            }

            Bp += static_cast<size_t>(2) * N;
        }

        if (k < K) {
            const vfloat32m2_t bv =
                __riscv_vle32_v_f32m2(Bp, vl);

            acc0  = __riscv_vfmacc_vf_f32m2(acc0,  A0[k],  bv, vl);
            acc1  = __riscv_vfmacc_vf_f32m2(acc1,  A1[k],  bv, vl);
            acc2  = __riscv_vfmacc_vf_f32m2(acc2,  A2[k],  bv, vl);
            acc3  = __riscv_vfmacc_vf_f32m2(acc3,  A3[k],  bv, vl);
            acc4  = __riscv_vfmacc_vf_f32m2(acc4,  A4[k],  bv, vl);
            acc5  = __riscv_vfmacc_vf_f32m2(acc5,  A5[k],  bv, vl);
            acc6  = __riscv_vfmacc_vf_f32m2(acc6,  A6[k],  bv, vl);
            acc7  = __riscv_vfmacc_vf_f32m2(acc7,  A7[k],  bv, vl);
            acc8  = __riscv_vfmacc_vf_f32m2(acc8,  A8[k],  bv, vl);
            acc9  = __riscv_vfmacc_vf_f32m2(acc9,  A9[k],  bv, vl);
            acc10 = __riscv_vfmacc_vf_f32m2(acc10, A10[k], bv, vl);
            acc11 = __riscv_vfmacc_vf_f32m2(acc11, A11[k], bv, vl);
            acc12 = __riscv_vfmacc_vf_f32m2(acc12, A12[k], bv, vl);
            acc13 = __riscv_vfmacc_vf_f32m2(acc13, A13[k], bv, vl);
        }

        __riscv_vse32_v_f32m2(C0  + col, acc0,  vl);
        __riscv_vse32_v_f32m2(C1  + col, acc1,  vl);
        __riscv_vse32_v_f32m2(C2  + col, acc2,  vl);
        __riscv_vse32_v_f32m2(C3  + col, acc3,  vl);
        __riscv_vse32_v_f32m2(C4  + col, acc4,  vl);
        __riscv_vse32_v_f32m2(C5  + col, acc5,  vl);
        __riscv_vse32_v_f32m2(C6  + col, acc6,  vl);
        __riscv_vse32_v_f32m2(C7  + col, acc7,  vl);
        __riscv_vse32_v_f32m2(C8  + col, acc8,  vl);
        __riscv_vse32_v_f32m2(C9  + col, acc9,  vl);
        __riscv_vse32_v_f32m2(C10 + col, acc10, vl);
        __riscv_vse32_v_f32m2(C11 + col, acc11, vl);
        __riscv_vse32_v_f32m2(C12 + col, acc12, vl);
        __riscv_vse32_v_f32m2(C13 + col, acc13, vl);

        col += static_cast<int>(vl);
    }
}


static inline void matmul_rows_by_14_unroll2_rvv(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = 0;

    while (row + 14 <= M) {
        matmul_m14_shared_b_unroll2_rvv(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );

        row += 14;
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


__attribute__((noinline))
static void matmul_m14_shared_b_unroll4_rvv(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    const float* A0  = A + static_cast<size_t>(0)  * K;
    const float* A1  = A + static_cast<size_t>(1)  * K;
    const float* A2  = A + static_cast<size_t>(2)  * K;
    const float* A3  = A + static_cast<size_t>(3)  * K;
    const float* A4  = A + static_cast<size_t>(4)  * K;
    const float* A5  = A + static_cast<size_t>(5)  * K;
    const float* A6  = A + static_cast<size_t>(6)  * K;
    const float* A7  = A + static_cast<size_t>(7)  * K;
    const float* A8  = A + static_cast<size_t>(8)  * K;
    const float* A9  = A + static_cast<size_t>(9)  * K;
    const float* A10 = A + static_cast<size_t>(10) * K;
    const float* A11 = A + static_cast<size_t>(11) * K;
    const float* A12 = A + static_cast<size_t>(12) * K;
    const float* A13 = A + static_cast<size_t>(13) * K;

    float* C0  = C + static_cast<size_t>(0)  * N;
    float* C1  = C + static_cast<size_t>(1)  * N;
    float* C2  = C + static_cast<size_t>(2)  * N;
    float* C3  = C + static_cast<size_t>(3)  * N;
    float* C4  = C + static_cast<size_t>(4)  * N;
    float* C5  = C + static_cast<size_t>(5)  * N;
    float* C6  = C + static_cast<size_t>(6)  * N;
    float* C7  = C + static_cast<size_t>(7)  * N;
    float* C8  = C + static_cast<size_t>(8)  * N;
    float* C9  = C + static_cast<size_t>(9)  * N;
    float* C10 = C + static_cast<size_t>(10) * N;
    float* C11 = C + static_cast<size_t>(11) * N;
    float* C12 = C + static_cast<size_t>(12) * N;
    float* C13 = C + static_cast<size_t>(13) * N;

    int col = 0;

    while (col < N) {
        const size_t vl =
            __riscv_vsetvl_e32m2(
                static_cast<size_t>(N - col)
            );

        vfloat32m2_t acc0  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc1  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc2  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc3  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc4  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc5  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc6  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc7  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc8  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc9  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc10 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc11 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc12 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t acc13 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        const float* Bp =
            B + static_cast<size_t>(col);

        int k = 0;

#define FMA14(KIDX, BPTR)                                                \
        do {                                                             \
            const vfloat32m2_t bv =                                     \
                __riscv_vle32_v_f32m2((BPTR), vl);                      \
                                                                         \
            acc0  = __riscv_vfmacc_vf_f32m2(acc0,  A0[(KIDX)],  bv, vl); \
            acc1  = __riscv_vfmacc_vf_f32m2(acc1,  A1[(KIDX)],  bv, vl); \
            acc2  = __riscv_vfmacc_vf_f32m2(acc2,  A2[(KIDX)],  bv, vl); \
            acc3  = __riscv_vfmacc_vf_f32m2(acc3,  A3[(KIDX)],  bv, vl); \
            acc4  = __riscv_vfmacc_vf_f32m2(acc4,  A4[(KIDX)],  bv, vl); \
            acc5  = __riscv_vfmacc_vf_f32m2(acc5,  A5[(KIDX)],  bv, vl); \
            acc6  = __riscv_vfmacc_vf_f32m2(acc6,  A6[(KIDX)],  bv, vl); \
            acc7  = __riscv_vfmacc_vf_f32m2(acc7,  A7[(KIDX)],  bv, vl); \
            acc8  = __riscv_vfmacc_vf_f32m2(acc8,  A8[(KIDX)],  bv, vl); \
            acc9  = __riscv_vfmacc_vf_f32m2(acc9,  A9[(KIDX)],  bv, vl); \
            acc10 = __riscv_vfmacc_vf_f32m2(acc10, A10[(KIDX)], bv, vl); \
            acc11 = __riscv_vfmacc_vf_f32m2(acc11, A11[(KIDX)], bv, vl); \
            acc12 = __riscv_vfmacc_vf_f32m2(acc12, A12[(KIDX)], bv, vl); \
            acc13 = __riscv_vfmacc_vf_f32m2(acc13, A13[(KIDX)], bv, vl); \
        } while (0)

        for (; k + 3 < K; k += 4) {
            FMA14(
                k,
                Bp
            );

            FMA14(
                k + 1,
                Bp + static_cast<size_t>(N)
            );

            FMA14(
                k + 2,
                Bp + static_cast<size_t>(2) * N
            );

            FMA14(
                k + 3,
                Bp + static_cast<size_t>(3) * N
            );

            Bp += static_cast<size_t>(4) * N;
        }

        for (; k < K; ++k) {
            FMA14(
                k,
                Bp
            );

            Bp += static_cast<size_t>(N);
        }

#undef FMA14

        __riscv_vse32_v_f32m2(C0  + col, acc0,  vl);
        __riscv_vse32_v_f32m2(C1  + col, acc1,  vl);
        __riscv_vse32_v_f32m2(C2  + col, acc2,  vl);
        __riscv_vse32_v_f32m2(C3  + col, acc3,  vl);
        __riscv_vse32_v_f32m2(C4  + col, acc4,  vl);
        __riscv_vse32_v_f32m2(C5  + col, acc5,  vl);
        __riscv_vse32_v_f32m2(C6  + col, acc6,  vl);
        __riscv_vse32_v_f32m2(C7  + col, acc7,  vl);
        __riscv_vse32_v_f32m2(C8  + col, acc8,  vl);
        __riscv_vse32_v_f32m2(C9  + col, acc9,  vl);
        __riscv_vse32_v_f32m2(C10 + col, acc10, vl);
        __riscv_vse32_v_f32m2(C11 + col, acc11, vl);
        __riscv_vse32_v_f32m2(C12 + col, acc12, vl);
        __riscv_vse32_v_f32m2(C13 + col, acc13, vl);

        col += static_cast<int>(vl);
    }
}



static inline void matmul_rows_by_14_unroll4_rvv(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = 0;

    while (row + 14 <= M) {
        matmul_m14_shared_b_unroll4_rvv(
            A + static_cast<size_t>(row) * K,
            B,
            C + static_cast<size_t>(row) * N,
            K,
            N
        );

        row += 14;
    }

    // M=1500 only leaves 2 rows
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



struct MatmulProfile {
    long long count = 0;
    double total_ms = 0.0;
};

static std::map<
    std::tuple<int, int, int>,
    MatmulProfile
> g_matmul_profiles;


static void print_matmul_profiles() {
    std::cerr << "\n===== MATMUL PROFILE =====\n";

    for (const auto& item : g_matmul_profiles) {
        const auto& key = item.first;
        const auto& p   = item.second;

        int M = std::get<0>(key);
        int K = std::get<1>(key);
        int N = std::get<2>(key);

        double avg_ms =
            p.count > 0
            ? p.total_ms / static_cast<double>(p.count)
            : 0.0;

        std::cerr
            << "M=" << M
            << " K=" << K
            << " N=" << N
            << " count=" << p.count
            << " total_ms=" << p.total_ms
            << " avg_ms=" << avg_ms
            << "\n";
    }

    std::cerr << "==========================\n";
}

struct MatmulProfilePrinter {
    ~MatmulProfilePrinter() {
        print_matmul_profiles();
    }
};

static MatmulProfilePrinter g_matmul_profile_printer;


static void matmul_m1_rvv_m8_range(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int start_col,
    int end_col
) {
    int col = start_col;

    while (col < end_col) {
        const size_t vl =
            __riscv_vsetvl_e32m8(
                static_cast<size_t>(end_col - col)
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

static void matmul_m1_rvv_m8_range_u2(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int start_col,
    int end_col
) {
    int col = start_col;

    while (col < end_col) {
        const size_t vl =
            __riscv_vsetvl_e32m8(
                static_cast<size_t>(end_col - col));

        vfloat32m8_t acc =
            __riscv_vfmv_v_f_f32m8(0.0f, vl);

        int k = 0;
        const float* Bp =
            B + static_cast<size_t>(col);

        for (; k + 1 < K; k += 2) {
            const vfloat32m8_t b0 =
                __riscv_vle32_v_f32m8(
                    Bp, vl);

            const vfloat32m8_t b1 =
                __riscv_vle32_v_f32m8(
                    Bp + static_cast<size_t>(N),
                    vl);

            acc =
                __riscv_vfmacc_vf_f32m8(
                    acc, A[k], b0, vl);

            acc =
                __riscv_vfmacc_vf_f32m8(
                    acc, A[k + 1], b1, vl);

            Bp += static_cast<size_t>(2) * N;
        }

        for (; k < K; ++k) {
            const vfloat32m8_t bv =
                __riscv_vle32_v_f32m8(
                    Bp, vl);

            acc =
                __riscv_vfmacc_vf_f32m8(
                    acc, A[k], bv, vl);

            Bp += N;
        }

        __riscv_vse32_v_f32m8(
            C + col, acc, vl);

        col += static_cast<int>(vl);
    }
}




static void matmul_m1_rvv_m8_range_u4(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int start_col,
    int end_col
){
    int col = start_col;

    while (col < end_col) {
        const size_t vl =
            __riscv_vsetvl_e32m8(
                static_cast<size_t>(
                    end_col - col));

        vfloat32m8_t acc =
            __riscv_vfmv_v_f_f32m8(
                0.0f, vl);

        int k = 0;

        const float* Bp =
            B + static_cast<size_t>(col);

        for (; k + 3 < K; k += 4) {
            const vfloat32m8_t b0 =
                __riscv_vle32_v_f32m8(
                    Bp, vl);

            const vfloat32m8_t b1 =
                __riscv_vle32_v_f32m8(
                    Bp + static_cast<size_t>(N),
                    vl);

            const vfloat32m8_t b2 =
                __riscv_vle32_v_f32m8(
                    Bp + static_cast<size_t>(2) * N,
                    vl);

            const vfloat32m8_t b3 =
                __riscv_vle32_v_f32m8(
                    Bp + static_cast<size_t>(3) * N,
                    vl);

            acc = __riscv_vfmacc_vf_f32m8(
                acc, A[k], b0, vl);

            acc = __riscv_vfmacc_vf_f32m8(
                acc, A[k + 1], b1, vl);

            acc = __riscv_vfmacc_vf_f32m8(
                acc, A[k + 2], b2, vl);

            acc = __riscv_vfmacc_vf_f32m8(
                acc, A[k + 3], b3, vl);

            Bp +=
                static_cast<size_t>(4) * N;
        }

        for (; k < K; ++k) {
            const vfloat32m8_t bv =
                __riscv_vle32_v_f32m8(
                    Bp, vl);

            acc = __riscv_vfmacc_vf_f32m8(
                acc, A[k], bv, vl);

            Bp += N;
        }

        __riscv_vse32_v_f32m8(
            C + col,
            acc,
            vl);

        col += static_cast<int>(vl);
    }
}



static void matmul_m1_rvv_m8_range_u8(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int start_col,
    int end_col
) {
    int col = start_col;

    while (col < end_col) {
        const size_t vl =
            __riscv_vsetvl_e32m8(
                static_cast<size_t>(end_col - col)
            );

        vfloat32m8_t acc =
            __riscv_vfmv_v_f_f32m8(
                0.0f,
                vl
            );

        int k = 0;

        const float* Bp =
            B + static_cast<size_t>(col);

        for (; k + 7 < K; k += 8) {
            const vfloat32m8_t b0 =
                __riscv_vle32_v_f32m8(
                    Bp,
                    vl
                );

            const vfloat32m8_t b1 =
                __riscv_vle32_v_f32m8(
                    Bp + static_cast<size_t>(N),
                    vl
                );

            const vfloat32m8_t b2 =
                __riscv_vle32_v_f32m8(
                    Bp + static_cast<size_t>(2) * N,
                    vl
                );

            const vfloat32m8_t b3 =
                __riscv_vle32_v_f32m8(
                    Bp + static_cast<size_t>(3) * N,
                    vl
                );

            const vfloat32m8_t b4 =
                __riscv_vle32_v_f32m8(
                    Bp + static_cast<size_t>(4) * N,
                    vl
                );

            const vfloat32m8_t b5 =
                __riscv_vle32_v_f32m8(
                    Bp + static_cast<size_t>(5) * N,
                    vl
                );

            const vfloat32m8_t b6 =
                __riscv_vle32_v_f32m8(
                    Bp + static_cast<size_t>(6) * N,
                    vl
                );

            const vfloat32m8_t b7 =
                __riscv_vle32_v_f32m8(
                    Bp + static_cast<size_t>(7) * N,
                    vl
                );

            acc = __riscv_vfmacc_vf_f32m8(
                acc, A[k], b0, vl);

            acc = __riscv_vfmacc_vf_f32m8(
                acc, A[k + 1], b1, vl);

            acc = __riscv_vfmacc_vf_f32m8(
                acc, A[k + 2], b2, vl);

            acc = __riscv_vfmacc_vf_f32m8(
                acc, A[k + 3], b3, vl);

            acc = __riscv_vfmacc_vf_f32m8(
                acc, A[k + 4], b4, vl);

            acc = __riscv_vfmacc_vf_f32m8(
                acc, A[k + 5], b5, vl);

            acc = __riscv_vfmacc_vf_f32m8(
                acc, A[k + 6], b6, vl);

            acc = __riscv_vfmacc_vf_f32m8(
                acc, A[k + 7], b7, vl);

            Bp += static_cast<size_t>(8) * N;
        }

        for (; k < K; ++k) {
            const vfloat32m8_t bv =
                __riscv_vle32_v_f32m8(
                    Bp,
                    vl
                );

            acc = __riscv_vfmacc_vf_f32m8(
                acc,
                A[k],
                bv,
                vl
            );

            Bp += N;
        }

        __riscv_vse32_v_f32m8(
            C + col,
            acc,
            vl
        );

        col += static_cast<int>(vl);
    }
}


static void matmul_m1_multithread(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    int T
) {
    T = std::max(1, std::min(T, 8));

    if (T == 1) {
        if (K == 384 && N == 1536) {
            matmul_m1_rvv_m8_range_u4(
                A, B, C,
                K, N,
                0, N
            );
        } else {
            matmul_m1_rvv_m8(
                A, B, C,
                K, N
            );
        }
        return;
    }

    std::vector<std::thread> threads;

    const int block = 64;
    const int full_blocks = N / block;
    const int tail = N % block;

    // 避免 thread 數比 block 還多
    T = std::min(T, std::max(1, full_blocks));

    const int q = full_blocks / T;
    const int r = full_blocks % T;

    int col = 0;

    for (int t = 0; t < T; ++t) {
        const int begin = col;

        const int blocks =
            q + (t < r ? 1 : 0);

        int cols = blocks * block;

        if (t == T - 1) {
            cols += tail;
        }

        const int end = begin + cols;
        col = end;

        threads.emplace_back([=]() {
            if (K == 384 && N == 1536) {
                matmul_m1_rvv_m8_range_u4(
                    A, B, C,
                    K, N,
                    begin, end
                );
            } /*else if (K == 1536 && N == 384) {
                matmul_m1_rvv_m8_range_u2(
                    A, B, C,
                    K, N,
                    begin, end
                );
            }*/ else {
                matmul_m1_rvv_m8_range(
                    A, B, C,
                    K, N,
                    begin, end
                );
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }
}

static int get_m1_threads(int K, int N) {
    if (K == 384 && N == 51865) {
        return 8;
    }

    if (K == 384 && N == 1536) {
        return 4;
    }

    if (K == 1536 && N == 384) {
        return 2;
    }

    if (K == 384 && N == 384) {
        return 1;
    }

    return 1;
}


static void matmul_m1500_multithread(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N
) {
    int T = 8;

    if (const char* s = std::getenv("KIWIPEDIA_MATMUL_THREADS")) {
        int requested = std::atoi(s);
        if (requested > 0) {
            T = requested;
        }
    }

    T = std::max(1, std::min(T, 8));
    std::vector<std::thread> threads;

    int tile;

    if ((K == 64 && N == 1500) ||
        (K == 384 && N == 1536) ||
        (K == 384 && N == 384)) {
        tile = 14;
    } else if (K == 1500 && N == 64) {
        tile = 10;
    } else if (K == 1536 && N == 384) {
        tile = 12;
    }  else if (K == 1152 && N == 384) {
        tile = 12;
    }else {
        matmul_rows_by_8_rvv(A, B, C, 1500, K, N);
        return;
    }

    const int M = 1500;

    int full_tiles = M / tile;
    int tail = M % tile;

    int q = full_tiles / T;
    int r = full_tiles % T;

    int row = 0;

    for (int t = 0; t < T; ++t) {
        int begin = row;

        int tiles_for_thread = q + (t < r ? 1 : 0);
        int rows = tiles_for_thread * tile;

        if (t == T - 1) {
            rows += tail;
        }

    row += rows;

        threads.emplace_back([=]() {
            const float* At = A + static_cast<size_t>(begin) * K;
            float* Ct = C + static_cast<size_t>(begin) * N;

            if ((K == 64 && N == 1500) ||
                (K == 384 && N == 1536)) {
                matmul_rows_by_14_unroll4_rvv(
                    At, B, Ct, rows, K, N);
            }
            else if (K == 384 && N == 384) {
                matmul_rows_by_14_unroll2_rvv(
                    At, B, Ct, rows, K, N);
            }
            else if (K == 1500 && N == 64) {
                matmul_rows_by_10_unroll4_rvv(
                    At, B, Ct, rows, K, N);
            }
            else if ((K == 1536 && N == 384) ||
                    (K == 1152 && N == 384)) {
                matmul_rows_by_12_unroll2_rvv(
                    At, B, Ct, rows, K, N);
            }
        });
    }

    for (auto& th : threads)
        th.join();
}



static void do_block_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    if (M == 1) {
        std::lock_guard<std::mutex> lock(seen_shapes_mutex);

        auto key = std::make_tuple(M, K, N);

        if (seen_shapes.insert(key).second) {
            std::cerr << "[MATMUL SHAPE] "
                      << "M=" << M
                      << " K=" << K
                      << " N=" << N
                      << std::endl;
        }
    }
    auto start = std::chrono::steady_clock::now();

    const bool use_m14_u4 =
        M == 1500 &&
        (
            (K == 64  && N == 1500) ||
            (K == 384 && N == 1536)
        );

    const bool use_m14_u2 =
        M == 1500 &&
        (
            K == 384 && N == 384
        );

    const bool use_m10_u4 =
        M == 1500 &&
        (
            K == 1500 && N == 64
        );

    const bool use_m12_u2 =
        M == 1500 &&
        (
            K == 1536 && N == 384
        );
    if (M == 1) {
        int T = get_m1_threads(K, N);

        matmul_m1_multithread(
            A, B, C,
            K, N, T
        );
    }
    else if (M == 1500) {
        matmul_m1500_multithread(
            A, B, C,
            K, N
        );
    }
    else {
    if (use_m14_u4) {
        matmul_rows_by_14_unroll4_rvv(
            A, B, C,
            M, K, N
        );
    } else if (use_m14_u2) {
        matmul_rows_by_14_unroll2_rvv(
            A, B, C,
            M, K, N
        );
    } else if (use_m10_u4) {
        matmul_rows_by_10_unroll4_rvv(
            A, B, C,
            M, K, N
        );
    } else if (use_m12_u2) {
        matmul_rows_by_12_unroll2_rvv(
            A, B, C,
            M, K, N
        );
    } else {
        matmul_rows_by_8_rvv(
            A, B, C,
            M, K, N
        );
    }
}
    auto end = std::chrono::steady_clock::now();

    double ms =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    auto key =
        std::make_tuple(M, K, N);

    auto& profile =
        g_matmul_profiles[key];

    profile.count += 1;
    profile.total_ms += ms;

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

