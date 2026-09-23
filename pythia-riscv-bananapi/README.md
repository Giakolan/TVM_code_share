# Pythia-70M RISC-V Banana Pi Deployment

Pythia-70M + TVM Relax BYOC + Kiwipedia + RISC-V RVV deployment package.

## Kernels

- libmatmul_original_scalar.so
  Original i-j-k scalar baseline

- libmatmul_scalar_baseline.so
  Optimized scalar v2/v4, auto-vectorization disabled

- libmatmul_rvv.so
  RVV optimized version

## Model

model/pythia_70m_riscv.so

## Runner

bin/pythia_runner_riscv
