# Pythia-70M M=1 Attention RVV Benchmark

Target:
- Banana Pi BPI-F3
- SpacemiT K1
- RISC-V + RVV

此 benchmark 用來比較 Pythia-70M Dynamic KV Cache Decode 階段的 M=1 Attention MatMul。

## Versions

- Generic: 原始 Generic RVV MatMul
- M1_v1: M=1 specialized MatMul
- M1_v2: M=1 specialized + local restrict + GCC RVV optimization

## Attention Shapes

QK:

```text
M = 1
K = 64
N = T
M = 1
K = T
N = 64
T = 16, 32, 64, 128, 256, 512, 1024
chmod +x benchmark_m1_attention_riscv
./benchmark_m1_attention_riscv \
  ./libmatmul_rvv.so \
  Generic \
  2000 \
| tee generic_bpi_f3.csv
./benchmark_m1_attention_riscv \
  ./libmatmul_m1_v1_bench_rvv.so \
  M1_v1 \
  2000 \
| tee m1_v1_bpi_f3.csv
./benchmark_m1_attention_riscv \
  ./libmatmul_m1_v2_bench_rvv.so \
  M1_v2 \
  2000 \
| tee m1_v2_bpi_f3.csv
kernel,mode,T,K,N,repeat,avg_us,max_diff

確認整個資料夾：

```bash
ls -lh pythia-70m/benchmarks/banana_pi_m1/
