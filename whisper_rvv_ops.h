#ifndef WHISPER_RVV_OPS_H
#define WHISPER_RVV_OPS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* 回傳 0：成功；-1：空指標、形狀、epsilon 或大小非法。
 * FP32、連續布局；呼叫端負責實際 buffer 容量與有限數輸入。
 * 允許 x == y；禁止部分重疊。gamma/beta 不得與輸出重疊。
 * 零元素是合法 no-op。沒有配置 heap memory，也不擁有輸入。
 */
int whisper_softmax_f32(const float* x, float* y, int64_t rows, int64_t cols);
/* Encoder 常見 [B*6,1500,1500]：rows = B*6*1500, cols = 1500。 */
int whisper_encoder_softmax_f32(const float* x, float* y, int64_t batch,
                                int64_t heads, int64_t query_len, int64_t key_len);
int whisper_layernorm_f32(const float* x, const float* gamma, const float* beta,
                          float* y, int64_t rows, int64_t hidden, float epsilon);
int whisper_gelu_f32(const float* x, float* y, int64_t count);
/* 1=真正編譯 RVV intrinsics；0=可攜式近似算法，不能視為 RVV 效能。 */
int whisper_has_rvv(void);
#ifdef __cplusplus
}
#endif
#endif
