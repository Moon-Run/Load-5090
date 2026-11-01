# RTX 5090 OP2：A/B/C 叠加 MA89 与 MA11

协议 `rtx5090-phase6-op2-abc-ma11-v1`；1 trial × 2 blocks × 3 组，共 6 窗口、300 个正式 optimizer step。

2 块短测已完成计划窗口，但未完成六块排序平衡周期；仅报告描述统计与配对点估计，不计算 95% CI。

## compare1

| 组别 | 平均 ms/step | 中位数 | 最慢 | tokens/s | 相对 A＋MA89＋MA11 降幅 | 最大 allocated GiB | allocated＋native GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A＋MA89＋MA11 | 861.225 | 861.225 | 861.425 | 9512.03 | +0.000% | 19.411822 | 19.411822 |
| B＋MA89＋MA11 | 994.713 | 994.713 | 998.847 | 8235.54 | -15.500% | 19.432940 | 19.432940 |
| C＋MA89＋MA11 | 1001.993 | 1001.993 | 1006.530 | 8175.70 | -16.345% | 20.281085 | 20.281085 |

| 配对（前者 / 后者） | 降幅 % [95% CI] | 更快块数 |
| --- | ---: | ---: |
| A＋MA89＋MA11 / B＋MA89＋MA11 | +13.419 ［2 块短测，不计算 CI］ | 2/2 |
| A＋MA89＋MA11 / C＋MA89＋MA11 | +14.048 ［2 块短测，不计算 CI］ | 2/2 |
| B＋MA89＋MA11 / C＋MA89＋MA11 | +0.726 ［2 块短测，不计算 CI］ | 2/2 |

全部窗口算术均值；配对估计使用完整块对数比。保留负收益。2 块短测不计算置信区间。
allocated＋native 不含 CUDA context、NCCL 等内部占用；逻辑贡献字节数不是 NCCL 实测链路流量。
各组独立通过数值和完整训练检查；不要求不同量化或归约的轨迹逐位一致。
数值观测 `flux-basic-interleaved/flux-basic-arrival`：loss 最大绝对差 0；gradient norm 最大绝对差 0。
数值观测 `flux-basic-interleaved/flux-basic-selective`：loss 最大绝对差 1.66732；gradient norm 最大绝对差 11.9129。
数值观测 `flux-basic-arrival/flux-basic-selective`：loss 最大绝对差 1.66732；gradient norm 最大绝对差 11.9129。

Three ABC projection paths with the same MA8 cache, MA9 zero statistic and MA11 softmax. MA11 already includes MA89: shared improvements are installed once. No CR3 consumer fusion is transplanted. Original ABC codec, chunks and CUDA connections stay fixed. No contemporaneous unoptimized or MA89-only control: this run cannot isolate their incremental gains. No historical pooling; all windows and negative gains retained. 2-block short run has incomplete counterbalancing and no confidence interval.
