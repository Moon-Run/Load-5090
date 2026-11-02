# RTX 5090 开源通信压缩三组对比

协议 `rtx5090-open-compress-int8-three-arm-v1`；1 trials × 1 blocks；150 正式 optimizer steps。

到达优先、选择性量化均关闭；全部计时窗口保留。

**单批探索性结果：不足六个完整块，不生成置信区间。**

## compare1

| 组别 | 平均 ms/step | 中位数 | 最慢 | tokens/s | 最大 allocated GiB | 最大 reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flux（BF16） | 985.922 | 985.922 | 985.922 | 8308.97 | 17.4908 | 17.6484 |
| Flux＋block INT8 | 949.537 | 949.537 | 949.537 | 8627.36 | 18.4182 | 18.6094 |
| Flux＋block INT8＋交替排序 | 948.592 | 948.592 | 948.592 | 8635.95 | 18.4182 | 18.6094 |

| 配对（前者/后者） | 均值比降幅 % | 配对对数比降幅 % | 配对 95% CI | 更快块 |
| --- | ---: | ---: | --- | ---: |
| flux-int8/flux-bf16 | +3.690 | +3.690 | 不估计 | 1/1 |
| flux-int8-interleaved/flux-int8 | +0.099 | +0.099 | 不估计 | 1/1 |
| flux-int8-interleaved/flux-bf16 | +3.786 | +3.786 | 不估计 | 1/1 |

算子、分段 CUDA event 与同步 wall time、显存见 `operator-summary.json`；原始证据见 `operator/`。
Torch allocated/reserved 分开报告；codec 无外部 cudaMalloc；整卡峰值仍不含 NCCL/context 的精确归因。

Fixed synthetic GPT6.7B workload. BF16 uses NCCL ReduceScatter; INT8 uses byte AllToAll and source-ordered BF16 reduction. INT8/BF16 is a whole-path comparison. Interleaving changes GEMM CTA mapping and can reassign StreamK reductions; no bitwise trajectory equivalence is claimed. No tile-arrival overlap, arrival policy, selective quantization or convergence claim. Payload bytes are not measured NCCL link traffic. CI only with >=6 blocks; comparisons unadjusted.
