# RTX 5090 OP4：NCCL 16 通道候选组

仅测试 OP4 候选：原 OP3 组合路径，设置 `NCCL_MIN_NCHANNELS=16` 和 `NCCL_MAX_NCHANNELS=16`。

1 trial × 2 个候选窗口，共 100 个正式 optimizer step；每窗口 10 步预热＋50 步正式计时。

| trial | 平均 ms/step | 中位数 | 最快 | 最慢 | tokens/s | 历史参考差值 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compare1 | 832.249 | 832.249 | 831.011 | 833.486 | 9843.21 | 17.946 |

历史参考为 850.195077378 ms；表中差值仅供观察，不是同期净收益。
全模型 smoke 梯度与历史参考的哈希允许不同，差异见 `candidate-evidence.json`；哈希无法表明数值差异大小。

各子批保留完整算子报告、原始窗口、显存和 trace；执行路径审计见 `candidate-evidence.json`。
没有同期对照，`target_validated=false`；completed 不代表验证了 60ms 提升。

Only the OP4 candidate is measured. No concurrent control, paired gain, bootstrap confidence interval or validated 60 ms improvement is available. The read-only reference is historical and any difference is observational, not a causal performance gain. Full-model smoke gradient hashes may differ from the historical reference; hash differences do not measure their numerical magnitude. Operator timing does not measure a complete training step.
