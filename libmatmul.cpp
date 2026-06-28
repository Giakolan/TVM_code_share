#include <vector>
#include <cstdint>
#include <dlpack/dlpack.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>


static int64_t NumElements(const DLTensor* t) {
    int64_t total = 1;
    for (int i = 0; i < t->ndim; ++i) {
        total *= t->shape[i];
    }
    return total;
}

static void PrintTensorStats(const char* name, const DLTensor* t) {
    std::cout << "[DEBUG][matmul] " << name << " shape=(";
    for (int i = 0; i < t->ndim; ++i) {
        std::cout << t->shape[i];
        if (i + 1 < t->ndim) std::cout << ", ";
    }
    std::cout << ") ";

    const float* p = static_cast<const float*>(t->data);
    int64_t total = NumElements(t);

    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    int64_t nan_count = 0;
    int64_t inf_count = 0;

    for (int64_t i = 0; i < total; ++i) {
        float v = p[i];
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (std::isinf(v)) {
            inf_count++;
            continue;
        }
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }

    std::cout << "numel=" << total
              << " min=" << mn
              << " max=" << mx
              << " nan=" << nan_count
              << " inf=" << inf_count
              << std::endl;
}
// -------------------- classic --------------------
// core kernel, operates on one [n, m] x [m, o] -> [n, o]

void matmul_classic(const float* A, const float* B, float* C,
                   int n, int m, int o) {
    for (int i = 0; i < n; ++i) {
        //std::cout<<i<<std::endl;
        for (int j = 0; j < o; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < m; ++k) {
                sum += A[i * m + k] * B[k * o + j];
            }
            //std::cout<<sum<<std::endl;
            C[i * o + j] = sum;
        }
    }
}

// -------------------- Batch x Batch --------------------
void matmul_bxb(std::vector<const DLTensor*>& data_entry_,
                int n, int m, int o, int batch) {
    //std::cout<<"batch"<<std::endl;
 
    const float* A = static_cast<const float*>(data_entry_[0]->data);
    const float* B = static_cast<const float*>(data_entry_[1]->data);
    float*       C = static_cast<float*>(data_entry_[2]->data);
    for (int bidx = 0; bidx < batch; ++bidx) {
        //std::cout<<bidx<<std::endl;
        const float* Ab = A + bidx * n * m;
        const float* Bb = B + bidx * m * o;
        float*       Cb = C + bidx * n * o;
        matmul_classic(Ab, Bb, Cb, n, m, o);
    }
}

// -------------------- Batch x Single --------------------
void matmul_bxs(std::vector<const DLTensor*>& data_entry_,
                int n, int m, int o, int batch) {
    const float* A = static_cast<const float*>(data_entry_[0]->data);
    const float* B = static_cast<const float*>(data_entry_[1]->data);
    float*       C = static_cast<float*>(data_entry_[2]->data);
    //std::cout<<"single"<<std::endl;
    for (int bidx = 0; bidx < batch; ++bidx) {
        //std::cout<<bidx<<std::endl;

        const float* Ab = A + bidx * n * m;

        float*       Cb = C + bidx * n * o;
        matmul_classic(Ab, B, Cb, n, m, o);
    }
}

// -------------------- Wrapper --------------------
extern "C"
void matmul(std::vector<const DLTensor*>& data_entry_,
            std::vector<int64_t>& shapeA,
            std::vector<int64_t>& shapeB) {
    static int call_id = 0;
    call_id++;

    const DLTensor* A = data_entry_[0];
    const DLTensor* B = data_entry_[1];
    const DLTensor* C = data_entry_[2];

    //std::cout << "\n========== [DEBUG][matmul call "
    //          << call_id << "] ==========" << std::endl;

    //std::cout << "[DEBUG][matmul] compile-time shapeA = ";
    //for (auto v : shapeA) std::cout << v << " ";
    //std::cout << std::endl;

    //std::cout << "[DEBUG][matmul] compile-time shapeB = ";
    //for (auto v : shapeB) std::cout << v << " ";
    //std::cout << std::endl;

    //PrintTensorStats("A before", A);
    //PrintTensorStats("B before", B);
    //PrintTensorStats("C before", C);

    auto check_output_size = [&](int64_t expected) -> bool {
        int64_t actual = NumElements(C);
        if (actual != expected) {
            std::cout << "[ERROR][matmul] output size mismatch. expected="
                      << expected << " actual=" << actual << std::endl;

            std::cout << "[ERROR][matmul] A ndim=" << A->ndim
                      << " B ndim=" << B->ndim
                      << " C ndim=" << C->ndim << std::endl;

            //PrintTensorStats("A before", A);
            //PrintTensorStats("B before", B);
            //PrintTensorStats("C before", C);

            std::cout << "[ERROR][matmul] Skip unsupported matmul to avoid memory overwrite."
                      << std::endl;

            return false;
        }
        return true;
    };

    if (A->ndim == 3 && B->ndim == 2) {
        // Supported:
        // A: [batch, n, m]
        // B: [m, o]
        // C: [batch, n, o]

        int batch = static_cast<int>(A->shape[0]);
        int n     = static_cast<int>(A->shape[1]);
        int m     = static_cast<int>(A->shape[2]);
        int o     = static_cast<int>(B->shape[1]);

        //std::cout << "[DEBUG][matmul] mode=bxs "
        //          << "batch=" << batch
        //          << " n=" << n
        //          << " m=" << m
        //          << " o=" << o << std::endl;

        int64_t expected = 1LL * batch * n * o;
        if (!check_output_size(expected)) return;

        matmul_bxs(data_entry_, n, m, o, batch);

    } else if (A->ndim == 3 && B->ndim == 3) {
        // Supported only when:
        // A: [batch, n, m]
        // B: [batch, m, o]
        // C: [batch, n, o]
        //
        // Some attention-related layouts do NOT match this,
        // so output size must be checked before computing.

        int batch = static_cast<int>(A->shape[0]);
        int n     = static_cast<int>(A->shape[1]);
        int m     = static_cast<int>(A->shape[2]);
        int o     = static_cast<int>(B->shape[2]);

        //std::cout << "[DEBUG][matmul] mode=bxb "
        //          << "batch=" << batch
        //          << " n=" << n
        //          << " m=" << m
        //          << " o=" << o << std::endl;

        int64_t expected = 1LL * batch * n * o;
        if (!check_output_size(expected)) return;

        matmul_bxb(data_entry_, n, m, o, batch);

    } else if (A->ndim == 2 && B->ndim == 2) {
        // Supported:
        // A: [n, m]
        // B: [m, o]
        // C: [n, o]

        int n = static_cast<int>(A->shape[0]);
        int m = static_cast<int>(A->shape[1]);
        int o = static_cast<int>(B->shape[1]);

        //std::cout << "[DEBUG][matmul] mode=2d "
        //          << "n=" << n
        //          << " m=" << m
        //          << " o=" << o << std::endl;

        int64_t expected = 1LL * n * o;
        if (!check_output_size(expected)) return;

        const float* a = static_cast<const float*>(A->data);
        const float* b = static_cast<const float*>(B->data);
        float* c = static_cast<float*>(C->data);

        matmul_classic(a, b, c, n, m, o);

    } else {
        std::cout << "[ERROR][matmul] unsupported ndim. A->ndim="
                  << A->ndim << " B->ndim=" << B->ndim
                  << " C->ndim=" << C->ndim << std::endl;

        //PrintTensorStats("A before", A);
        //PrintTensorStats("B before", B);
        //PrintTensorStats("C before", C);

        return;
    }

    //PrintTensorStats("C after", C);

    //std::cout << "========== [DEBUG][matmul call "
    //          << call_id << " end] ==========\n" << std::endl;
}

extern "C" void add(std::vector<const DLTensor*>& data_entry_,
                    std::vector<int64_t>& A_shape,
                    std::vector<int64_t>& B_shape) {
    const DLTensor* A = data_entry_[0];
    const DLTensor* B = data_entry_[1];
    const DLTensor* C = data_entry_[2];

    const float* a = static_cast<const float*>(A->data);
    const float* b = static_cast<const float*>(B->data);
    float* c = static_cast<float*>(C->data);

    int64_t total = 1;
    for (int i = 0; i < C->ndim; ++i) {
        total *= C->shape[i];
    }

    for (int64_t i = 0; i < total; ++i) {
        c[i] = a[i] + b[i];
    }
}
