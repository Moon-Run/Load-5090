# RTX 5090 OP1 A/B/C 与 Flux 改进量化完整训练对比

协议 `rtx5090-phase6-op1-abc-staged-v1`；1 trial × 2 blocks × 4 组，共 8 窗口、400 个正式 optimizer step。

2 块短测已完成计划窗口，但未完成八块排序平衡周期；仅报告描述统计与配对点估计，不计算 95% CI。

## compare1

| 组别 | 平均 ms/step | 中位数 | 最慢 | tokens/s | 相对改进量化降幅 | 最大 allocated GiB | allocated＋native GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flux＋基础量化＋交替排序（A） | 966.738 | 966.738 | 966.754 | 8473.85 | +1.928% | 18.412770 | 18.412770 |
| Flux＋基础量化＋交替排序＋到达优先（B） | 1077.683 | 1077.683 | 1083.984 | 7601.49 | -9.327% | 18.448872 | 18.448872 |
| Flux＋基础量化＋交替排序＋到达优先＋选择性量化（C） | 1074.402 | 1074.402 | 1076.957 | 7624.71 | -8.994% | 19.274037 | 19.274037 |
| Flux＋改进量化 | 985.740 | 985.740 | 985.846 | 8310.51 | +0.000% | 17.678090 | 20.334431 |

| 配对（前者 / 后者） | 降幅 % [95% CI] | 更快块数 |
| --- | ---: | ---: |
| Flux＋基础量化＋交替排序（A） / Flux＋基础量化＋交替排序＋到达优先（B） | +10.293 ［2 块短测，不计算 CI］ | 2/2 |
| Flux＋基础量化＋交替排序（A） / Flux＋基础量化＋交替排序＋到达优先＋选择性量化（C） | +10.021 ［2 块短测，不计算 CI］ | 2/2 |
| Flux＋基础量化＋交替排序（A） / Flux＋改进量化 | +1.928 ［2 块短测，不计算 CI］ | 2/2 |
| Flux＋基础量化＋交替排序＋到达优先（B） / Flux＋基础量化＋交替排序＋到达优先＋选择性量化（C） | -0.304 ［2 块短测，不计算 CI］ | 1/2 |
| Flux＋基础量化＋交替排序＋到达优先（B） / Flux＋改进量化 | -9.325 ［2 块短测，不计算 CI］ | 0/2 |
| Flux＋基础量化＋交替排序＋到达优先＋选择性量化（C） / Flux＋改进量化 | -8.994 ［2 块短测，不计算 CI］ | 0/2 |

全部窗口算术均值；配对估计使用完整块对数比。保留负收益。2 块短测不计算置信区间。
allocated＋native 不含 CUDA context、NCCL 等内部占用；逻辑贡献字节数不是 NCCL 实测链路流量。
各组独立通过数值和完整训练检查；不要求不同量化或归约的轨迹逐位一致。
数值观测 `flux-basic-interleaved/flux-basic-arrival`：loss 最大绝对差 0；gradient norm 最大绝对差 0。
数值观测 `flux-basic-interleaved/flux-basic-selective`：loss 最大绝对差 1.66732；gradient norm 最大绝对差 11.9129。
数值观测 `flux-basic-interleaved/flux-improved`：loss 最大绝对差 4.25119；gradient norm 最大绝对差 56.2691。
数值观测 `flux-basic-arrival/flux-basic-selective`：loss 最大绝对差 1.66732；gradient norm 最大绝对差 11.9129。
数值观测 `flux-basic-arrival/flux-improved`：loss 最大绝对差 4.25119；gradient norm 最大绝对差 56.2691。
数值观测 `flux-basic-selective/flux-improved`：loss 最大绝对差 2.8838；gradient norm 最大绝对差 49.7163。

Fixed single-node RTX5090 TP8 workload. A exchanges one complete packet; B/C exchange four output-row waves and enqueue each decoder after its NCCL completion event. GEMM completes before communication; no original cross-GPU tile-arrival overlap. C quantizes a deterministic last 1/64 of groups and keeps the remainder BF16, without measured 5090 arrival calibration. Flux improved retains the separately documented original-order fused implementation. GEMM, codec, decoder, transport and rounding can differ; gains are end-to-end combinations. No historical pooling or filtering of negative gains. The 2-block short run does not complete the balanced ordering cycle; paired point estimates only, without confidence intervals.
