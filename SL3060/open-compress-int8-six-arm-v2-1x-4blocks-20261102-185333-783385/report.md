# RTX 5090 开源通信压缩六组对比

协议 `rtx5090-open-compress-int8-six-arm-v2`；1 trials × 4 blocks；1200 正式 optimizer steps。

到达优先、选择性量化均关闭；全部计时窗口保留。

**单批探索性结果：不足六个完整块，不生成置信区间。**

## compare1

| 组别 | 平均 ms/step | 中位数 | 最慢 | tokens/s | 最大 allocated GiB | 最大 reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flux（BF16） | 985.764 | 985.730 | 986.352 | 8310.31 | 17.4908 | 17.6484 |
| Flux＋block INT8 | 946.932 | 947.219 | 947.378 | 8651.10 | 18.4182 | 18.6094 |
| Flux＋block INT8＋交替排序 | 947.390 | 947.335 | 948.389 | 8646.91 | 18.4182 | 18.6094 |
| Megatron 原生（BF16） | 994.677 | 994.703 | 994.911 | 8235.84 | 16.3660 | 16.5273 |
| Megatron＋block INT8 | 955.821 | 956.230 | 956.635 | 8570.64 | 17.2784 | 17.5273 |
| Flux＋block INT8＋远端优先 | 946.006 | 946.413 | 947.232 | 8659.57 | 18.4182 | 18.6094 |

| 配对（前者/后者） | 均值比降幅 % | 配对对数比降幅 % | 配对 95% CI | 更快块 |
| --- | ---: | ---: | --- | ---: |
| flux-int8/flux-bf16 | +3.939 | +3.939 | 不估计 | 4/4 |
| flux-int8-interleaved/flux-int8 | -0.048 | -0.048 | 不估计 | 1/4 |
| flux-int8-remote-first/flux-int8 | +0.098 | +0.098 | 不估计 | 3/4 |
| flux-int8-remote-first/flux-int8-interleaved | +0.146 | +0.146 | 不估计 | 4/4 |
| megatron-int8/megatron-bf16 | +3.906 | +3.906 | 不估计 | 4/4 |
| flux-bf16/megatron-bf16 | +0.896 | +0.896 | 不估计 | 4/4 |
| flux-int8/megatron-int8 | +0.930 | +0.930 | 不估计 | 4/4 |
| flux-bf16/flux-int8-interleaved | -4.050 | -4.050 | 不估计 | 0/4 |
| flux-bf16/megatron-int8 | -3.133 | -3.133 | 不估计 | 0/4 |
| flux-bf16/flux-int8-remote-first | -4.203 | -4.203 | 不估计 | 0/4 |
| flux-int8/megatron-bf16 | +4.800 | +4.800 | 不估计 | 4/4 |
| flux-int8-interleaved/megatron-bf16 | +4.754 | +4.754 | 不估计 | 4/4 |
| flux-int8-interleaved/megatron-int8 | +0.882 | +0.882 | 不估计 | 4/4 |
| megatron-bf16/flux-int8-remote-first | -5.145 | -5.145 | 不估计 | 0/4 |
| megatron-int8/flux-int8-remote-first | -1.038 | -1.038 | 不估计 | 0/4 |

算子、分段 CUDA event 与同步 wall time、显存见 `operator-summary.json`；原始证据见 `operator/`。
Torch allocated/reserved 分开报告；codec 无外部 cudaMalloc；整卡峰值仍不含 NCCL/context 的精确归因。

Fixed synthetic GPT6.7B workload. BF16 uses NCCL ReduceScatter; INT8 uses byte AllToAll and source-ordered BF16 reduction. Native Megatron retains its RowParallelLinear forward and linear/collective backward; only the INT8 forward ReduceScatter boundary is substituted. INT8/BF16 is a whole-path comparison. Interleaving and remote-first change GEMM CTA mapping and can reassign StreamK reductions; no bitwise trajectory equivalence is claimed. No tile-arrival overlap, arrival policy, selective quantization or convergence claim. Payload bytes are not measured NCCL link traffic. CI only with >=6 blocks; comparisons unadjusted.
