#include <vector>
#include <cstdint>
#include <dlpack/dlpack.h>
#include <iostream>
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
    int n, m, o, x;
    // [n,m]*[m,o]
    //std::cout<<"matmul() starts"<<std::endl;

    if (shapeB.size() == 3) {
        x = shapeA[0];
        n = shapeA[1];
        m = shapeA[2];
        o = shapeB[2];
        matmul_bxb(data_entry_, n, m, o, x);
    } else {
        x = shapeA[0];
        n = shapeA[1];
        m = shapeA[2];
        o = shapeB[1];
        matmul_bxs(data_entry_, n, m, o, x);
    }
    //std::cout<<"matmul() ends"<<std::endl;

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
