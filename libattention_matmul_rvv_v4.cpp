#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <dlpack/dlpack.h>
#include <riscv_vector.h>

// Whisper Tiny Attention RVV v4
//
// New in v4:
//   1) Fixed Cross-QK specialization for BH=6, M=1, K=64, N=1500.
//   2) On VLEN=128/FP32, fixed execution plan:
//        23 x dual-m8 blocks (1472 columns) + 16/8/4 tail.
//   3) Persistent BH=6 thread pool for 2/3/6-thread head-level parallelism.
//   4) ATTENTION_V4_THREADS can override the integration-time thread count.
//
// v3 dynamic QK/PV and generic fallback paths are retained for all other shapes.

// Whisper Tiny Attention RVV v4
//
// Input layout (same as v1/v2):
//   A: [BH, M, K]
//   B: [BH, K, N]
//   C: [BH, M, N]
//
// inherited v3 fast paths:
//   QK: M=1, K=64, N=dynamic KV
//   PV: M=1, N=64, K=dynamic KV
//
// v3 ideas retained in v4:
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
// v4 Cross-Attention QK specialization
// Fixed hotspot shape:
//   BH=6, M=1, K=64, N=1500
//
// v4a: one-core / serial BH=6 execution.
// v4b: persistent worker pool for BH=6 head-level parallel execution.
//
// On VLEN=128 + FP32:
//   VLMAX(e32,m8) = 32
//   two m8 accumulators cover 64 columns per K traversal
//   1500 = 23 * 64 + 28
//   fixed tail = 16 (m4) + 8 (m2) + 4 (m1)
// -----------------------------------------------------------------------------

constexpr int kCrossBH = 6;
constexpr int kCrossK = 64;
constexpr int kCrossN = 1500;

// Compute one Cross-Attention head.  This is the v4a core kernel.
// The VLEN=128 path is deliberately hard-coded for the measured hotspot.
// A portable fallback preserves correctness on a different VLEN.
__attribute__((noinline))
static void qk_cross1500_head_v4(
    const float* __restrict__ q,
    const float* __restrict__ kt,
    float* __restrict__ out) {
    const int vlmax8 = static_cast<int>(__riscv_vsetvlmax_e32m8());
    const int vlmax4 = static_cast<int>(__riscv_vsetvlmax_e32m4());
    const int vlmax2 = static_cast<int>(__riscv_vsetvlmax_e32m2());
    const int vlmax1 = static_cast<int>(__riscv_vsetvlmax_e32m1());

    // Fastest intended configuration: VLEN=128, FP32.
    if (vlmax8 == 32 && vlmax4 == 16 && vlmax2 == 8 && vlmax1 == 4) {
        // Main region: exactly 23 dual-m8 blocks = 23 * 64 = 1472 columns.
        // N and K are compile-time constants in this path.
        for (int block = 0; block < 23; ++block) {
            const int col = block * 64;
            qk_dual_m8_k64(q, kt, out, kCrossN, col, 32);
        }

        // Fixed 28-column tail: 16 + 8 + 4.
        qk_range_m4_k64(q, kt, out, kCrossN, 1472, 16);
        qk_range_m2_k64(q, kt, out, kCrossN, 1488, 8);
        qk_range_m1_k64(q, kt, out, kCrossN, 1496, 4);
        return;
    }

    // Portable fallback for another VLEN.  It still keeps N=1500 and K=64
    // fixed, but determines the vector-group widths from the implementation.
    int col = 0;
    const int dual_cols = 2 * vlmax8;

    while (kCrossN - col >= dual_cols) {
        qk_dual_m8_k64(q, kt, out, kCrossN, col, vlmax8);
        col += dual_cols;
    }

    if (col < kCrossN) {
        qk_hierarchical_tail_k64(q, kt, out, kCrossN, col);
    }
}

static inline void qk_cross1500_bh6_serial_v4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {
    for (int b = 0; b < kCrossBH; ++b) {
        const float* __restrict__ q =
            A + static_cast<size_t>(b) * kCrossK;
        const float* __restrict__ kt =
            B + static_cast<size_t>(b) * kCrossK * kCrossN;
        float* __restrict__ out =
            C + static_cast<size_t>(b) * kCrossN;

        qk_cross1500_head_v4(q, kt, out);
    }
}

// Persistent BH=6 thread pool.  Threads are created once on the first parallel
// call and reused by subsequent decoder steps, avoiding per-call thread-create
// and join overhead.  Each worker owns independent attention heads, so no
// output synchronization is needed inside the kernel itself.
class CrossQK1500ThreadPool {
public:
    CrossQK1500ThreadPool() {
        workers_.reserve(kCrossBH);
        for (int worker_id = 0; worker_id < kCrossBH; ++worker_id) {
            workers_.emplace_back([this, worker_id]() {
                worker_loop(worker_id);
            });
        }
    }

    ~CrossQK1500ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
            ++generation_;
        }
        cv_work_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    CrossQK1500ThreadPool(const CrossQK1500ThreadPool&) = delete;
    CrossQK1500ThreadPool& operator=(const CrossQK1500ThreadPool&) = delete;

    void run(
        const float* A,
        const float* B,
        float* C,
        int threads) {
        // The pool is shared by all calls to this translation unit.  Serialize
        // dispatches so two concurrent callers cannot overwrite one job state.
        std::lock_guard<std::mutex> dispatch_guard(dispatch_mu_);

        threads = std::max(1, std::min(kCrossBH, threads));
        if (threads == 1) {
            qk_cross1500_bh6_serial_v4(A, B, C);
            return;
        }

        std::unique_lock<std::mutex> lock(mu_);
        A_ = A;
        B_ = B;
        C_ = C;
        active_threads_ = threads;
        pending_workers_ = threads;
        ++generation_;

        cv_work_.notify_all();
        cv_done_.wait(lock, [this]() {
            return pending_workers_ == 0;
        });
    }

private:
    void worker_loop(int worker_id) {
        std::uint64_t seen_generation = 0;

        for (;;) {
            const float* A = nullptr;
            const float* B = nullptr;
            float* C = nullptr;
            int active_threads = 1;

            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_work_.wait(lock, [this, &seen_generation]() {
                    return stop_ || generation_ != seen_generation;
                });

                if (stop_) return;

                seen_generation = generation_;
                A = A_;
                B = B_;
                C = C_;
                active_threads = active_threads_;
            }

            // Workers outside the selected thread count stay idle for this job.
            if (worker_id >= active_threads) {
                continue;
            }

            // Cyclic head assignment keeps the work balanced:
            //   2T: worker0 -> heads 0,2,4; worker1 -> 1,3,5
            //   3T: worker0 -> 0,3; worker1 -> 1,4; worker2 -> 2,5
            //   6T: one head per worker.
            for (int b = worker_id; b < kCrossBH; b += active_threads) {
                const float* q =
                    A + static_cast<size_t>(b) * kCrossK;
                const float* kt =
                    B + static_cast<size_t>(b) * kCrossK * kCrossN;
                float* out =
                    C + static_cast<size_t>(b) * kCrossN;

                qk_cross1500_head_v4(q, kt, out);
            }

            {
                std::lock_guard<std::mutex> lock(mu_);
                --pending_workers_;
                if (pending_workers_ == 0) {
                    cv_done_.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex dispatch_mu_;
    std::mutex mu_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;

    bool stop_ = false;
    std::uint64_t generation_ = 0;
    const float* A_ = nullptr;
    const float* B_ = nullptr;
    float* C_ = nullptr;
    int active_threads_ = 1;
    int pending_workers_ = 0;
};

static CrossQK1500ThreadPool& cross_qk1500_pool() {
    static CrossQK1500ThreadPool pool;
    return pool;
}

static int default_cross_qk_threads() {
    // Allow explicit tuning on Banana Pi / QEMU:
    //   ATTENTION_V4_THREADS=1|2|3|6
    if (const char* s = std::getenv("ATTENTION_V4_THREADS")) {
        const int requested = std::atoi(s);
        if (requested > 0) {
            return std::max(1, std::min(kCrossBH, requested));
        }
    }

    const unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) return 1;
    return std::max(1, std::min(kCrossBH, static_cast<int>(hc)));
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

// -----------------------------------------------------------------------------
// Public v4 benchmark entry points
// -----------------------------------------------------------------------------

extern "C" void attention_qk_cross1500_v4_serial_f32(
    const float* A,
    const float* B,
    float* C) {
    qk_cross1500_bh6_serial_v4(A, B, C);
}

extern "C" void attention_qk_cross1500_v4_parallel_f32(
    const float* A,
    const float* B,
    float* C,
    int threads) {
    threads = std::max(1, std::min(kCrossBH, threads));
    if (threads == 1) {
        qk_cross1500_bh6_serial_v4(A, B, C);
        return;
    }
    cross_qk1500_pool().run(A, B, C, threads);
}

extern "C" void attention_bmm_rvv_f32(
    const float* A,
    const float* B,
    float* C,
    int BH,
    int M,
    int K,
    int N) {
    // v4 Cross-Attention hotspot: [6,1,64] x [6,64,1500].
    // Use N=1500 specialization plus BH=6 persistent head-level parallelism.
    if (BH == kCrossBH && M == 1 && K == kCrossK && N == kCrossN) {
        const int threads = default_cross_qk_threads();
        if (threads <= 1) {
            qk_cross1500_bh6_serial_v4(A, B, C);
        } else {
            cross_qk1500_pool().run(A, B, C, threads);
        }
        return;
    }

    // Other Whisper decoder QK shapes retain the v3 dynamic-KV path.
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
        std::cerr << "[ATTENTION_RVV_V4] expected A, B, C\n";
        return;
    }

    const DLTensor* A_t = data_entry_[0];
    const DLTensor* B_t = data_entry_[1];
    const DLTensor* C_t = data_entry_[2];

    if (A_t->ndim != 3 || B_t->ndim != 3 || C_t->ndim != 3) {
        std::cerr << "[ATTENTION_RVV_V4] only 3D BMM supported\n";
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
        std::cerr << "[ATTENTION_RVV_V4] shape mismatch\n";
        return;
    }

    const float* A = static_cast<const float*>(A_t->data);
    const float* B = static_cast<const float*>(B_t->data);
    float* C = static_cast<float*>(C_t->data);

    attention_bmm_rvv_f32(A, B, C, BH, M, K, N);
}
