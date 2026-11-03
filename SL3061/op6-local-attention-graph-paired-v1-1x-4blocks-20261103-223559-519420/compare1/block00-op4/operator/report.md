# OP3 组合组：独立投影算子

A＋MA89＋MA11＋R＋CR3 消费者融合＋OP3 编码器改进

每角色 20 次预热＋1 个窗口×100 次调用；每窗口分别取八 rank 最大 CUDA-event/同步墙钟平均延迟。
正确性阶段检查融合/非融合输出、四类梯度和三步更新逐位一致；只有组合路径进入计时。

| 角色 | GPU 平均 μs | GPU 中位 μs | GPU 最慢窗口 μs | 墙钟平均 μs |
| --- | ---: | ---: | ---: | ---: |
| attn | 628.976 | 628.976 | 628.976 | 629.446 |
| mlp | 463.542 | 463.542 | 463.542 | 463.756 |

独立 profile：先取每 rank 的调用中位数，再取八 rank 中位数。

| 角色 | GEMM μs | 编码 μs | NCCL kernel 合计 μs | 解码含融合 μs | CPU 提交 μs | GPU 起止跨度 μs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attn | 41.072 | 26.384 | 1374.264 | 8.928 | 869.310 | 1454.981 |
| mlp | 147.983 | 26.256 | 265.156 | 8.920 | 390.738 | 452.530 |

Preallocated inputs and one persistent operator per role. No LayerNorm, backward, optimizer, MA8/MA9/MA11 work or CUDA Graph in the timing interval. Forward/backward and three updates are separately checked against unfused autograd. Profile kernel times are diagnostics, not additive full-step savings.
