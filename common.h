#ifndef WHISPER_COMMON_H
#define WHISPER_COMMON_H
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include "whisper_rvv_ops.h"
#if defined(__riscv_vector) && !defined(WHISPER_FORCE_SCALAR)
#define W_RVV 1
#include <riscv_vector.h>
#else
#define W_RVV 0
#endif
#if defined(WHISPER_REQUIRE_RVV) && !W_RVV
#error "RVV intrinsics required: check target flags and compiler"
#endif
static inline bool valid_shape(int64_t rows, int64_t cols) {
    return rows >= 0 && cols > 0 && (uint64_t)rows <= (uint64_t)PTRDIFF_MAX / sizeof(float) / (uint64_t)cols;
}
/* exp(x), x<=0：range reduction + 七次 Taylor polynomial。
 * 小於 -87 直接歸零；省略量至多 exp(-87)，不追求 subnormal 相對精度。
 * ln2 拆成兩段，避免大負數時相減損失精度。
 */
static inline float exp_negative_scalar(float x) {
    if (x < -87.0f) return 0.0f;
    x = fminf(x, 0.0f);
    int32_t n = (int32_t)fmaf(x, 1.4426950408889634f, -0.5f);
    float r = fmaf(-(float)n, 0.693359375f, x);
    r = fmaf(-(float)n, -0.00021219444005469058f, r);
    float p = 1.0f / 5040.0f;
    p = fmaf(p,r,1.0f/720.0f); p = fmaf(p,r,1.0f/120.0f);
    p = fmaf(p,r,1.0f/24.0f); p = fmaf(p,r,1.0f/6.0f);
    p = fmaf(p,r,0.5f); p = fmaf(p,r,1.0f); p = fmaf(p,r,1.0f);
    uint32_t bits = (uint32_t)(n + 127) << 23;
    float scale; memcpy(&scale,&bits,sizeof(scale));
    return p * scale;
}
/* Abramowitz & Stegun 7.1.26 的 erfc 近似形式。
 * 負 x 直接用 erfc 避免 1-erf 的消去誤差；不是 tanh GELU。
 */
static inline float gelu_scalar(float x) {
    float a = fminf(fabsf(x), 14.0f) * 0.7071067811865475f;
    float t = 1.0f / fmaf(0.3275911f, a, 1.0f);
    float p = 1.061405429f;
    p = fmaf(p,t,-1.453152027f); p = fmaf(p,t,1.421413741f);
    p = fmaf(p,t,-0.284496736f); p = fmaf(p,t,0.254829592f);
    float tail = 0.5f * p * t * exp_negative_scalar(-a*a);
    return x * (x < 0.0f ? tail : 1.0f-tail);
}
#if W_RVV
static inline vfloat32m1_t exp_negative_vec(vfloat32m1_t x, size_t vl) {
    vbool32_t under = __riscv_vmflt_vf_f32m1_b32(x,-87.0f,vl);
    x = __riscv_vfmax_vf_f32m1(x,-87.0f,vl);
    x = __riscv_vfmin_vf_f32m1(x,0.0f,vl);
    vfloat32m1_t q = __riscv_vfmv_v_f_f32m1(-0.5f,vl);
    q = __riscv_vfmacc_vf_f32m1(q,1.4426950408889634f,x,vl);
    vint32m1_t n = __riscv_vfcvt_rtz_x_f_v_i32m1(q,vl);
    vfloat32m1_t nf = __riscv_vfcvt_f_x_v_f32m1(n,vl);
    vfloat32m1_t r = __riscv_vfmacc_vf_f32m1(x,-0.693359375f,nf,vl);
    r = __riscv_vfmacc_vf_f32m1(r,0.00021219444005469058f,nf,vl);
    vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(1.0f/5040.0f,vl);
#define W_HORNER(c) p = __riscv_vfmadd_vv_f32m1(p,r,__riscv_vfmv_v_f_f32m1(c,vl),vl)
    W_HORNER(1.0f/720.0f); W_HORNER(1.0f/120.0f); W_HORNER(1.0f/24.0f);
    W_HORNER(1.0f/6.0f); W_HORNER(0.5f); W_HORNER(1.0f); W_HORNER(1.0f);
#undef W_HORNER
    n = __riscv_vadd_vx_i32m1(n,127,vl);
    n = __riscv_vsll_vx_i32m1(n,23,vl);
    p = __riscv_vfmul_vv_f32m1(p,__riscv_vreinterpret_v_i32m1_f32m1(n),vl);
    return __riscv_vfmerge_vfm_f32m1(p,0.0f,under,vl);
}
static inline vfloat32m1_t gelu_vec(vfloat32m1_t x, size_t vl) {
    vfloat32m1_t a = __riscv_vfabs_v_f32m1(x,vl);
    a = __riscv_vfmin_vf_f32m1(a,14.0f,vl);
    a = __riscv_vfmul_vf_f32m1(a,0.7071067811865475f,vl);
    vfloat32m1_t d = __riscv_vfmv_v_f_f32m1(1.0f,vl);
    d = __riscv_vfmacc_vf_f32m1(d,0.3275911f,a,vl);
    vfloat32m1_t t = __riscv_vfrdiv_vf_f32m1(d,1.0f,vl);
    vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(1.061405429f,vl);
#define W_HORNER(c) p = __riscv_vfmadd_vv_f32m1(p,t,__riscv_vfmv_v_f_f32m1(c,vl),vl)
    W_HORNER(-1.453152027f); W_HORNER(1.421413741f);
    W_HORNER(-0.284496736f); W_HORNER(0.254829592f);
#undef W_HORNER
    vfloat32m1_t e = __riscv_vfmul_vv_f32m1(a,a,vl);
    e = exp_negative_vec(__riscv_vfneg_v_f32m1(e,vl),vl);
    p = __riscv_vfmul_vv_f32m1(p,t,vl);
    p = __riscv_vfmul_vv_f32m1(p,e,vl);
    p = __riscv_vfmul_vf_f32m1(p,0.5f,vl);
    vfloat32m1_t positive = __riscv_vfrsub_vf_f32m1(p,1.0f,vl);
    vbool32_t negative = __riscv_vmflt_vf_f32m1_b32(x,0.0f,vl);
    p = __riscv_vmerge_vvm_f32m1(positive,p,negative,vl);
    return __riscv_vfmul_vv_f32m1(x,p,vl);
}
#endif
#endif
