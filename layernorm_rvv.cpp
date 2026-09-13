#include "common.h"
static void layernorm_row(const float* x,const float* gamma,const float* beta,
                          float* y,int64_t hidden,float eps) {
    double sum=0.0, sqsum=0.0;
#if W_RVV
    // 加總使用 widening FP64 reduction，減輕均值累積誤差。
    vfloat64m1_t seed=__riscv_vfmv_v_f_f64m1(0.0,1);
    for(int64_t j=0;j<hidden;) {
        size_t vl=__riscv_vsetvl_e32m1((size_t)(hidden-j));
        auto v=__riscv_vle32_v_f32m1(x+j,vl);
        seed=__riscv_vfwredusum_vs_f32m1_f64m1(v,seed,vl);
        j+=(int64_t)vl;
    }
    sum=__riscv_vfmv_f_s_f64m1_f64(seed);
#else
    for(int64_t j=0;j<hidden;++j) sum+=x[j];
#endif
    float mean=(float)(sum/(double)hidden);
#if W_RVV
    seed=__riscv_vfmv_v_f_f64m1(0.0,1);
    for(int64_t j=0;j<hidden;) {
        size_t vl=__riscv_vsetvl_e32m1((size_t)(hidden-j));
        auto v=__riscv_vle32_v_f32m1(x+j,vl);
        v=__riscv_vfsub_vf_f32m1(v,mean,vl);
        v=__riscv_vfmul_vv_f32m1(v,v,vl);
        seed=__riscv_vfwredusum_vs_f32m1_f64m1(v,seed,vl);
        j+=(int64_t)vl;
    }
    sqsum=__riscv_vfmv_f_s_f64m1_f64(seed);
#else
    for(int64_t j=0;j<hidden;++j) { float d=x[j]-mean; sqsum+=(double)(d*d); }
#endif
    float inv=1.0f/sqrtf((float)(sqsum/(double)hidden)+eps);
#if W_RVV
    for(int64_t j=0;j<hidden;) {
        size_t vl=__riscv_vsetvl_e32m1((size_t)(hidden-j));
        auto v=__riscv_vle32_v_f32m1(x+j,vl);
        v=__riscv_vfsub_vf_f32m1(v,mean,vl);
        v=__riscv_vfmul_vf_f32m1(v,inv,vl);
        auto g=__riscv_vle32_v_f32m1(gamma+j,vl);
        auto b=__riscv_vle32_v_f32m1(beta+j,vl);
        v=__riscv_vfmacc_vv_f32m1(b,v,g,vl);
        __riscv_vse32_v_f32m1(y+j,v,vl);
        j+=(int64_t)vl;
    }
#else
    for(int64_t j=0;j<hidden;++j) y[j]=fmaf((x[j]-mean)*inv,gamma[j],beta[j]);
#endif
}
extern "C" int whisper_layernorm_f32(const float* x,const float* gamma,const float* beta,
                                      float* y,int64_t rows,int64_t hidden,float epsilon) {
    if(!valid_shape(rows,hidden)||!(epsilon>0.0f)||!isfinite(epsilon)) return -1;
    if(rows==0) return 0;
    if(!x||!y||!gamma||!beta) return -1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(rows>=64 && rows*hidden>=32768)
#endif
    for(int64_t r=0;r<rows;++r) layernorm_row(x+r*hidden,gamma,beta,y+r*hidden,hidden,epsilon);
    return 0;
}
