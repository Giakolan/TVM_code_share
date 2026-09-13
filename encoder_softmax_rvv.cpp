#include "common.h"
static void softmax_row(const float* x, float* y, int64_t cols) {
    float max_value = -INFINITY;
#if W_RVV
    vfloat32m1_t max_seed = __riscv_vfmv_v_f_f32m1(-INFINITY,1);
    for (int64_t j=0;j<cols;) {
        size_t vl=__riscv_vsetvl_e32m1((size_t)(cols-j));
        auto v=__riscv_vle32_v_f32m1(x+j,vl);
        max_seed=__riscv_vfredmax_vs_f32m1_f32m1(v,max_seed,vl);
        j+=(int64_t)vl;
    }
    max_value=__riscv_vfmv_f_s_f32m1_f32(max_seed);
    vfloat32m1_t sum_seed=__riscv_vfmv_v_f_f32m1(0.0f,1);
    for (int64_t j=0;j<cols;) {
        size_t vl=__riscv_vsetvl_e32m1((size_t)(cols-j));
        auto v=__riscv_vle32_v_f32m1(x+j,vl);
        v=exp_negative_vec(__riscv_vfsub_vf_f32m1(v,max_value,vl),vl);
        __riscv_vse32_v_f32m1(y+j,v,vl);
        sum_seed=__riscv_vfredusum_vs_f32m1_f32m1(v,sum_seed,vl);
        j+=(int64_t)vl;
    }
    float inv=1.0f/__riscv_vfmv_f_s_f32m1_f32(sum_seed);
    for (int64_t j=0;j<cols;) {
        size_t vl=__riscv_vsetvl_e32m1((size_t)(cols-j));
        auto v=__riscv_vle32_v_f32m1(y+j,vl);
        __riscv_vse32_v_f32m1(y+j,__riscv_vfmul_vf_f32m1(v,inv,vl),vl);
        j+=(int64_t)vl;
    }
#else
    for(int64_t j=0;j<cols;++j) max_value=fmaxf(max_value,x[j]);
    double sum=0;
    for(int64_t j=0;j<cols;++j) { y[j]=exp_negative_scalar(x[j]-max_value);sum+=y[j]; }
    float inv=(float)(1.0/sum);
    for(int64_t j=0;j<cols;++j) y[j]*=inv;
#endif
}
extern "C" int whisper_softmax_f32(const float* x,float* y,int64_t rows,int64_t cols) {
    if(!valid_shape(rows,cols)) return -1;
    if(rows==0) return 0;
    if(!x||!y) return -1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(rows>=64 && rows*cols>=32768)
#endif
    for(int64_t r=0;r<rows;++r) softmax_row(x+r*cols,y+r*cols,cols);
    return 0;
}
extern "C" int whisper_encoder_softmax_f32(const float* x,float* y,int64_t batch,
                                           int64_t heads,int64_t query_len,int64_t key_len) {
    if(batch<0||heads<=0||query_len<=0||key_len<=0) return -1;
    if(batch>INT64_MAX/heads) return -1;
    int64_t rows=batch*heads;
    if(rows>INT64_MAX/query_len) return -1;
    return whisper_softmax_f32(x,y,rows*query_len,key_len);
}
