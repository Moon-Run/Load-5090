# RTX 5090 Flux 四组量化对比

协议 `rtx5090-phase6-flux-quant-four-arm-v1`；1 trial × 8 blocks × 4 组，共 32 窗口、1600 个正式 optimizer step。

## compare1

| 组别 | 平均 ms/step | 中位数 | 最慢 | tokens/s | 相对无量化降幅 | 最大 allocated GiB | allocated＋native GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flux＋改进量化 | 1016.920 | 986.157 | 1111.599 | 8055.70 | +0.282% | 17.678090 | 20.334431 |
| Flux＋量化融合 | 1040.040 | 1033.086 | 1113.506 | 7876.62 | -1.985% | 17.008412 | 18.336583 |
| Flux＋基础量化 | 1029.783 | 1053.539 | 1090.423 | 7955.08 | -0.979% | 17.473256 | 17.473256 |
| Flux（无量化） | 1019.798 | 985.687 | 1107.685 | 8032.96 | +0.000% | 17.490834 | 17.490834 |

| 配对（前者 / 后者） | 降幅 % [95% CI] | 更快块数 |
| --- | ---: | ---: |
| Flux＋改进量化 / Flux＋量化融合 | +2.217 [-3.271, +6.796] | 5/8 |
| Flux＋改进量化 / Flux＋基础量化 | +1.258 [-2.480, +5.081] | 3/8 |
| Flux＋改进量化 / Flux（无量化） | +0.311 [-1.675, +2.765] | 2/8 |
| Flux＋量化融合 / Flux＋基础量化 | -0.981 [-4.621, +2.397] | 2/8 |
| Flux＋量化融合 / Flux（无量化） | -1.949 [-7.150, +2.873] | 4/8 |
| Flux＋基础量化 / Flux（无量化） | -0.959 [-4.429, +1.751] | 6/8 |

全部窗口算术均值；配对估计使用完整块对数比和 10,000 次 bootstrap。保留负收益。
allocated＋native 不含 CUDA context、NCCL 等内部占用；逻辑贡献字节数不是 NCCL 实测链路流量。
各组独立通过数值和完整训练检查；不要求不同量化或归约的轨迹逐位一致。
数值观测 `flux-improved/flux-fused`：loss 最大绝对差 0；gradient norm 最大绝对差 0。
数值观测 `flux-improved/flux-basic`：loss 最大绝对差 3.45332；gradient norm 最大绝对差 56.3494。
数值观测 `flux-improved/flux-bf16`：loss 最大绝对差 2.00606；gradient norm 最大绝对差 40.4213。
数值观测 `flux-fused/flux-basic`：loss 最大绝对差 3.45332；gradient norm 最大绝对差 56.3494。
数值观测 `flux-fused/flux-bf16`：loss 最大绝对差 2.00606；gradient norm 最大绝对差 40.4213。
数值观测 `flux-basic/flux-bf16`：loss 最大绝对差 3.62291；gradient norm 最大绝对差 96.7708。

Fixed single-node RTX5090 TP8 workload. Old C/A use original Flux rank-offset scheduling and full-remote quantization; no interleave, arrival calibration or selective mask. Unquantized Flux uses GemmOnly plus NCCL BF16 Reduce-Scatter without P2P. GEMM, decoder, transport and rounding differ across arms; gains are end-to-end combinations. No historical pooling or filtering of negative gains; six paired CIs are unadjusted.
