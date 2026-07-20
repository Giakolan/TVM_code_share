#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlpack/dlpack.h>
#include <riscv_vector.h>

// === Blocking tile size ===
#ifndef MC
#define MC 64
#endif
#ifndef NC
#define NC 128
#endif
#ifndef KC
#define KC 128
#endif

// ==================== 1. PACK_B (The Gap Remover) ====================
/**
 * Pack B tile into contiguous 1D buffer: [KC][NC] layout (row-major)
 * This ensures B[k][j+1] immediately follows B[k][j] in memory.
 * 
 * @param B: Source matrix (column-major storage with leading dimension O)
 * @param o: Number of columns in B
 * @param pc: Starting row index in B
 * @param kc: Number of rows to pack
 * @param jc: Starting column index in B
 * @param nc: Number of columns to pack
 * @param Bp: Destination packed buffer (must be at least kc*nc floats)
 */
static inline void pack_B_tile(
    const float* B, 
    int o,
    int pc, 
    int kc, 
    int jc, 
    int nc,
    float* Bp
) {
    // Pack as [KC][NC] - each "row" of the tile is NC elements wide
    for (int k = 0; k < kc; ++k) {
        // Source: Row (pc+k) of B, starting at column jc
        const float* B_row = B + (size_t)(pc + k) * (size_t)o + (size_t)jc;
        
        // Destination: Position k*nc in the packed buffer
        float* Bp_row = Bp + (size_t)k * (size_t)nc;
        
        // Contiguous copy - NO GAPS
        memcpy(Bp_row, B_row, (size_t)nc * sizeof(float));
    }
}

// ==================== 2. RVV MICROKERNEL (Unit-Stride Only) ====================
/**
 * Compute C[mc][nc] += A[mc][kc] * Bp[kc][nc] using RVV intrinsics
 * 
 * Key invariant: Bp is packed contiguously as [kc][nc], so:
 *   Bp[k*nc + j] gives element B[k][j]
 * 
 * @param Ablk: Pointer to A tile (row-major, leading dimension lda)
 * @param lda: Leading dimension of A (typically M)
 * @param Bp: Packed B tile [kc][nc] (contiguous)
 * @param Cblk: Pointer to C tile (row-major, leading dimension ldc)
 * @param ldc: Leading dimension of C (typically O)
 * @param mc: Number of rows in A tile
 * @param kc: Inner dimension
 * @param nc: Number of columns in B tile
 */
static inline void microkernel_rvv_unit_stride(
    const float* Ablk, 
    int lda,
    const float* Bp,
    float* Cblk, 
    int ldc,
    int mc, 
    int kc, 
    int nc
) {
    // Outer loop: rows of A (and C)
    for (int i = 0; i < mc; ++i) {
        const float* A_row = Ablk + (size_t)i * (size_t)lda;
        float* C_row = Cblk + (size_t)i * (size_t)ldc;
        
        // Middle loop: vectorize across columns of B (and C)
        int col = 0;
        while (col < nc) {
            // Set vector length for remaining columns
            size_t vl = __riscv_vsetvl_e32m1((size_t)(nc - col));
            
            // Load accumulator from C (to support accumulation across K-tiles)
            vfloat32m1_t vacc = __riscv_vle32_v_f32m1(C_row + col, vl);
            
            // Inner loop: reduction over K
            for (int k = 0; k < kc; ++k) {
                // Broadcast A[i][k]
                float a_scalar = A_row[k];
                
                // UNIT-STRIDE LOAD: B[k][col:col+vl]
                // Position in Bp: k*nc + col
                const float* B_vec = Bp + (size_t)k * (size_t)nc + (size_t)col;
                vfloat32m1_t bv = __riscv_vle32_v_f32m1(B_vec, vl);
                
                // FMA: vacc += a_scalar * bv
                vacc = __riscv_vfmacc_vf_f32m1(vacc, a_scalar, bv, vl);
            }
            
            // Write back to C
            __riscv_vse32_v_f32m1(C_row + col, vacc, vl);
            col += (int)vl;
        }
    }
}

// ==================== 3. BLOCKED MATMUL (3-Loop Tiling) ====================
/**
 * Blocked matrix multiplication: C = A * B
 * A: [n][m] row-major
 * B: [m][o] row-major  
 * C: [n][o] row-major
 */
void do_block_matmul(
    const float* A, 
    const float* B, 
    float* C,
    int n, 
    int m, 
    int o
) {
    // Aligned packing buffer (static to avoid repeated allocation)
    static float Bpack[KC * NC] __attribute__((aligned(64)));
    
    // Loop J: Tile columns of B (and C)
    for (int jc = 0; jc < o; jc += NC) {
        int nc = (jc + NC <= o) ? NC : (o - jc);
        
        // Loop I: Tile rows of A (and C)
        for (int ic = 0; ic < n; ic += MC) {
            int mc = (ic + MC <= n) ? MC : (n - ic);
            
            // Initialize C tile to zero (for accumulation across K-tiles)
            for (int i = 0; i < mc; ++i) {
                float* C_row = C + (size_t)(ic + i) * (size_t)o + (size_t)jc;
                memset(C_row, 0, (size_t)nc * sizeof(float));
            }
            
            // Loop P: Tile inner dimension K (and accumulate into C)
            for (int pc = 0; pc < m; pc += KC) {
                int kc = (pc + KC <= m) ? KC : (m - pc);
                
                // Pack B tile: [kc][nc] contiguous
                pack_B_tile(B, o, pc, kc, jc, nc, Bpack);
                
                // Compute: C[ic:ic+mc][jc:jc+nc] += A[ic:ic+mc][pc:pc+kc] * Bpack[kc][nc]
                const float* A_tile = A + (size_t)ic * (size_t)m + (size_t)pc;
                float* C_tile = C + (size_t)ic * (size_t)o + (size_t)jc;
                
                microkernel_rvv_unit_stride(
                    A_tile, m,      // A tile and its leading dimension
                    Bpack,          // Packed B
                    C_tile, o,      // C tile and its leading dimension
                    mc, kc, nc      // Tile dimensions
                );
            }
        }
    }
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

// Batch x Single: All batches share the same B
void matmul_bxs(
    std::vector<const DLTensor*>& data_entry_,
    int n, int m, int o, int batch
) {
    const float* A = static_cast<const float*>(data_entry_[0]->data);
    const float* B = static_cast<const float*>(data_entry_[1]->data);
    float* C = static_cast<float*>(data_entry_[2]->data);

    for (int b = 0; b < batch; ++b) {
        const float* A_batch = A + (size_t)b * (size_t)n * (size_t)m;
        float* C_batch = C + (size_t)b * (size_t)n * (size_t)o;
        
        do_block_matmul(A_batch, B, C_batch, n, m, o);
    }
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