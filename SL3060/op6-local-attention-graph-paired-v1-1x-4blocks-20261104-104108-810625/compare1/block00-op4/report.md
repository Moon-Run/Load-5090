# RTX 5090 OP3：编码器＋R＋CR3 融合组合组

协议 `rtx5090-phase6-op3-combined-v2`；1 trial × 1 blocks × 1 组，共 1 窗口、50 个正式 optimizer step。

每窗口独立启动，10 个 warmup＋50 个正式 step；只测组合组，不计算同期相对降幅或配对 CI。

| trial | 平均 ms/step | 中位数 | 最快 | 最慢 | tokens/s | 最大 allocated GiB | allocated＋native GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compare1 | 831.308 | 831.308 | 831.308 | 831.308 | 9854.34 | 19.422869 | 19.422869 |

算子延迟及阶段 profile 见 `operator/report.md`；算子时间不等于整步耗时。
allocated＋native 不含 CUDA context、NCCL 等内部占用。

Single combined configuration. All windows are retained. Operator forward/gradient equivalence is checked against the unfused autograd reference before measurement. No concurrent performance control, paired gain, paired confidence interval or historical pooling.
