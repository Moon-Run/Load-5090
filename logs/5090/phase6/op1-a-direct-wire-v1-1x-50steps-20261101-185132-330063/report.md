# RTX 5090 OP1-A 三组完整训练对比

协议 `rtx5090-phase6-op1-a-direct-wire-v1`；1 trial × 6 blocks × 3 组；18 个窗口、900 个正式 optimizer step。

全部窗口算术均值；每窗口 10 步预热＋50 步正式计时，取八 rank 最大墙钟。

## compare1

| 组别 | 平均 ms/step | 中位数 | 最慢窗口 | tokens/s | A 相对均值降幅 | 最大 allocated GiB | allocated＋native GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A（5090 direct-wire） | 978.184 | 978.640 | 979.205 | 8374.70 | +0.000% | 16.892811 | 18.220982 |
| Flux＋基础量化 | 967.327 | 967.481 | 968.949 | 8468.69 | -1.122% | 17.473256 | 17.473256 |
| Flux＋基础量化＋交替排序 | 967.302 | 967.512 | 969.549 | 8468.91 | -1.125% | 17.473256 | 17.473256 |

| 配对（前者相对后者） | 降幅 % [95% CI] | 更快块数 |
| --- | ---: | ---: |
| A（5090 direct-wire） / Flux＋基础量化 | -1.122 [-1.300, -0.985] | 0/6 |
| A（5090 direct-wire） / Flux＋基础量化＋交替排序 | -1.125 [-1.304, -0.971] | 0/6 |
| Flux＋基础量化＋交替排序 / Flux＋基础量化 | +0.003 [-0.059, +0.078] | 2/6 |

正降幅表示前者更快；保留负收益。配对结果与均值比是不同统计量。
allocated＋native 不含 CUDA context/NCCL 内部占用，不能当作整卡峰值。
A 使用不同 GEMM 实现；两个 Flux 组保持原 Stream-K，但排序可能改变归约分配和舍入，不要求逐位同轨迹。

Flux 两组数值观测：8 个配对窗口；loss 最大绝对差 1.06048，gradient norm 最大绝对差 4.30762；最终参数逐位相同的窗口 0/8。
这些是同场数值观测，不是模型收敛或质量等价结论。

Fixed single-node RTX5090 TP8 synthetic workload. No P2P or GEMM/NCCL overlap. A and Flux differ in GEMM implementation. The two Flux arms differ in CTA ordering, which may also change Stream-K reduction assignment and rounding; exact trajectories are not required. All windows, including negative gains, retained; paired bootstrap intervals are not adjusted for three comparisons. No historical windows are pooled.
