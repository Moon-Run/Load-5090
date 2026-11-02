# RTX 5090 OP3：R 方案与实测瓶颈改进方案

协议 `rtx5090-phase6-op3-two-arm-v1`；2 trial × 20 blocks × 2 组，共 80 窗口、4000 个正式 optimizer step。

每两块采用 AB/BA 顺序；每个窗口独立启动，均保留 10 个 warmup 与 50 个正式 step。

## compare1

| 组别 | 平均 ms/step | 中位数 | 最慢 | tokens/s | 相对 R 方案降幅 | 最大 allocated GiB | allocated＋native GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A＋MA89＋MA11＋R＋CR3 消费者融合 | 855.544 | 855.230 | 858.047 | 9575.19 | +0.000% | 19.422869 | 19.422869 |
| A＋MA89＋MA11＋OP3 改进 | 854.444 | 854.322 | 857.537 | 9587.53 | +0.129% | 19.411822 | 19.411822 |

| 配对（改进方案 / R 方案） | 降幅 % [95% CI] | 更快块数 |
| --- | ---: | ---: |
| A＋MA89＋MA11＋OP3 改进 / A＋MA89＋MA11＋R＋CR3 消费者融合 | +0.129 [+0.067, +0.191] | 19/20 |
## compare2

| 组别 | 平均 ms/step | 中位数 | 最慢 | tokens/s | 相对 R 方案降幅 | 最大 allocated GiB | allocated＋native GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A＋MA89＋MA11＋R＋CR3 消费者融合 | 855.393 | 855.318 | 857.690 | 9576.88 | +0.000% | 19.422869 | 19.422869 |
| A＋MA89＋MA11＋OP3 改进 | 855.132 | 854.721 | 857.011 | 9579.80 | +0.030% | 19.411822 | 19.411822 |

| 配对（改进方案 / R 方案） | 降幅 % [95% CI] | 更快块数 |
| --- | ---: | ---: |
| A＋MA89＋MA11＋OP3 改进 / A＋MA89＋MA11＋R＋CR3 消费者融合 | +0.031 [-0.024, +0.083] | 13/20 |

均值包含全部窗口；配对估计使用完整块的对数比。正值表示改进方案比 R 方案更快，保留负收益。
allocated＋native 不含 CUDA context、NCCL 等内部占用；逻辑贡献字节数不是 NCCL 实测链路流量。
两组裸投影的 packet 与输出必须通过逐位一致检查，源归约顺序不允许改变；smoke 的 loss、gradient norm、全部参数梯度与更新轨迹必须一致，随后计时窗口的训练轨迹作观测。
模型 smoke 中 R 方案记录融合 bias/residual 后的输出，改进方案记录裸投影；这两类输出哈希不直接比较。
数值观测 `op3-improved/op3-r`：loss 最大绝对差 0；gradient norm 最大绝对差 0。

Both arms share the same A projection foundation and one installation of MA8, MA9 and MA11. The registered per-arm definitions identify the R and OP3 changes. AB/BA order is balanced within every two blocks. All windows and negative gains are retained. This comparison estimates the difference between these two concurrent arms; it does not isolate the incremental gain over the historical OP2 baseline. No historical pooling. The paired interval uses 10,000 bootstrap samples.
