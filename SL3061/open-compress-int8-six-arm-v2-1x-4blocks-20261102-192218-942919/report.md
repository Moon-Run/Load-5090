# RTX 5090 开源通信压缩六组对比

协议 `rtx5090-open-compress-int8-six-arm-v2`；1 trials × 4 blocks；1200 正式 optimizer steps。

到达优先、选择性量化均关闭；全部计时窗口保留。

**单批探索性结果：不足六个完整块，不生成置信区间。**

## compare1

| 组别 | 平均 ms/step | 中位数 | 最慢 | tokens/s | 最大 allocated GiB | 最大 reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flux（BF16） | 868.832 | 869.581 | 871.106 | 9428.75 | 17.4908 | 17.6484 |
| Flux＋block INT8 | 868.987 | 869.955 | 877.890 | 9427.06 | 18.4182 | 18.6094 |
| Megatron 原生（BF16） | 876.301 | 875.591 | 881.995 | 9348.38 | 16.3660 | 16.5273 |
| Megatron＋block INT8 | 895.438 | 898.797 | 905.253 | 9148.60 | 17.2784 | 17.5273 |
| Flux＋block INT8＋远端优先 | 883.897 | 884.388 | 897.819 | 9268.05 | 18.4182 | 18.6094 |
| Flux＋block INT8＋交替排序 | 875.600 | 871.009 | 894.412 | 9355.87 | 18.4182 | 18.6094 |

| 配对（前者/后者） | 均值比降幅 % | 配对对数比降幅 % | 配对 95% CI | 更快块 |
| --- | ---: | ---: | --- | ---: |
| flux-int8/flux-bf16 | -0.018 | -0.014 | 不估计 | 2/4 |
| flux-int8-interleaved/flux-int8 | -0.761 | -0.757 | 不估计 | 1/4 |
| flux-int8-remote-first/flux-int8 | -1.716 | -1.713 | 不估计 | 1/4 |
| flux-int8-remote-first/flux-int8-interleaved | -0.948 | -0.949 | 不估计 | 1/4 |
| megatron-int8/megatron-bf16 | -2.184 | -2.178 | 不估计 | 0/4 |
| flux-bf16/megatron-bf16 | +0.852 | +0.852 | 不估计 | 4/4 |
| flux-int8/megatron-int8 | +2.954 | +2.952 | 不估计 | 4/4 |
| flux-bf16/flux-int8-interleaved | +0.773 | +0.765 | 不估计 | 2/4 |
| flux-bf16/megatron-int8 | +2.971 | +2.966 | 不估计 | 4/4 |
| flux-bf16/flux-int8-remote-first | +1.704 | +1.698 | 不估计 | 3/4 |
| flux-int8/megatron-bf16 | +0.835 | +0.838 | 不估计 | 2/4 |
| flux-int8-interleaved/megatron-bf16 | +0.080 | +0.087 | 不估计 | 3/4 |
| flux-int8-interleaved/megatron-int8 | +2.215 | +2.217 | 不估计 | 3/4 |
| megatron-bf16/flux-int8-remote-first | +0.859 | +0.853 | 不估计 | 3/4 |
| megatron-int8/flux-int8-remote-first | -1.306 | -1.306 | 不估计 | 0/4 |

算子、分段 CUDA event 与同步 wall time、显存见 `operator-summary.json`；原始证据见 `operator/`。
Torch allocated/reserved 分开报告；codec 无外部 cudaMalloc；整卡峰值仍不含 NCCL/context 的精确归因。

Fixed synthetic GPT6.7B workload. BF16 uses NCCL ReduceScatter; INT8 uses byte AllToAll and source-ordered BF16 reduction. Native Megatron retains its RowParallelLinear forward and linear/collective backward; only the INT8 forward ReduceScatter boundary is substituted. INT8/BF16 is a whole-path comparison. Interleaving and remote-first change GEMM CTA mapping and can reassign StreamK reductions; no bitwise trajectory equivalence is claimed. No tile-arrival overlap, arrival policy, selective quantization or convergence claim. Payload bytes are not measured NCCL link traffic. CI only with >=6 blocks; comparisons unadjusted.
