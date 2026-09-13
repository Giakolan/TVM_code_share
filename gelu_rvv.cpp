#include "common.h"
extern "C" int whisper_gelu_f32(const float* x,float* y,int64_t count) {
    if(!valid_shape(count,1)) return -1;
    if(count==0) return 0;
    if(!x||!y) return -1;
    const int64_t block=4096;
    int64_t blocks=count/block+(count%block!=0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count>=32768)
#endif
    for(int64_t b=0;b<blocks;++b) {
        int64_t begin=b*block, end=begin+((count-begin)<block?(count-begin):block);
#if W_RVV
        for(int64_t j=begin;j<end;) {
            size_t vl=__riscv_vsetvl_e32m1((size_t)(end-j));
            auto v=__riscv_vle32_v_f32m1(x+j,vl);
            __riscv_vse32_v_f32m1(y+j,gelu_vec(v,vl),vl);
            j+=(int64_t)vl;
        }
#else
        for(int64_t j=begin;j<end;++j) y[j]=gelu_scalar(x[j]);
#endif
    }
    return 0;
}
