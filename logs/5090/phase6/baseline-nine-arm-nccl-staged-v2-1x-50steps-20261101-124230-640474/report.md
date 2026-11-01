# RTX 5090 九组 NCCL staging v2 实测汇总

本协议不依赖 CUDA peer access；原 H100 跨卡 tile 到达和计算通信重叠不保留，不能与原协议混算。

完整 optimizer step；各组均使用全部正式窗口的算术均值。正降幅表示 MA11 更快。

协议：`rtx5090-phase6-nine-arm-nccl-staged-v2`；1 trial × 18 块 × 9 组；共 162 个计时窗口、8100 个正式 optimizer step。

## compare1

| 组别 | 平均 ms/step | 中位数 | 最慢窗口 | tokens/s | MA11 相对均值下降 | 最大 Torch allocated GiB | Torch 峰值＋TACO workspace GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MA11 | 911.458 | 911.603 | 912.119 | 8987.79 | +0.000% | 18.727984 | 21.384326 |
| MA89 | 943.601 | 943.621 | 943.930 | 8681.63 | +3.406% | 18.727984 | 21.384326 |
| MA8 | 965.172 | 965.721 | 966.544 | 8487.60 | +5.565% | 18.727776 | 21.384118 |
| MA9 | 995.407 | 995.406 | 995.997 | 8229.80 | +8.434% | 17.729867 | 20.386209 |
| CR3 | 1017.192 | 1017.265 | 1017.859 | 8053.54 | +10.395% | 17.729668 | 20.386010 |
| R | 977.081 | 977.195 | 978.014 | 8384.16 | +6.716% | 17.008412 | 18.336583 |
| C | 1022.978 | 1023.035 | 1024.961 | 8007.99 | +10.902% | 17.730156 | 20.386498 |
| A | 986.816 | 986.831 | 988.477 | 8301.44 | +7.636% | 17.008412 | 18.336583 |
| Flux＋基础量化 | 967.474 | 967.859 | 969.195 | 8467.41 | +5.790% | 17.473256 | 17.473256 |

显存合计逐 rank 相加后取最大值，计入独立 cudaMalloc 的 TACO workspace；仍不包含 CUDA context、NCCL 等内部显存，不能视为整卡总占用。

以下按完整块的配对对数耗时比估计，使用 10,000 次 bootstrap；与上表均值比是不同统计量。

| 同期配对（前者相对后者） | 配对降幅 %［95% CI］ | 完整块数 |
| --- | --- | ---: |
| MA11 / MA89 | +3.406［+3.384, +3.428］ | 18 |
| MA11 / MA8 | +5.565［+5.427, +5.649］ | 18 |
| MA11 / MA9 | +8.434［+8.410, +8.459］ | 18 |
| MA11 / CR3 | +10.395［+10.363, +10.421］ | 18 |
| MA11 / R | +6.716［+6.684, +6.748］ | 18 |
| MA11 / C | +10.901［+10.860, +10.946］ | 18 |
| MA11 / A | +7.636［+7.604, +7.674］ | 18 |
| MA11 / Flux＋基础量化 | +5.790［+5.686, +5.856］ | 18 |
| MA89 / MA8 | +2.235［+2.090, +2.317］ | 18 |
| MA89 / MA9 | +5.205［+5.190, +5.219］ | 18 |
| MA89 / CR3 | +7.235［+7.204, +7.261］ | 18 |
| MA89 / R | +3.426［+3.399, +3.455］ | 18 |
| MA89 / C | +7.759［+7.728, +7.793］ | 18 |
| MA89 / A | +4.379［+4.346, +4.414］ | 18 |
| MA89 / Flux＋基础量化 | +2.467［+2.366, +2.532］ | 18 |
| MA8 / MA9 | +3.038［+2.955, +3.182］ | 18 |
| MA8 / CR3 | +5.114［+5.029, +5.253］ | 18 |
| MA8 / R | +1.219［+1.131, +1.365］ | 18 |
| MA8 / C | +5.651［+5.559, +5.799］ | 18 |
| MA8 / A | +2.194［+2.118, +2.322］ | 18 |
| MA8 / Flux＋基础量化 | +0.238［+0.088, +0.420］ | 18 |
| MA9 / CR3 | +2.142［+2.105, +2.175］ | 18 |
| MA9 / R | -1.876［-1.907, -1.843］ | 18 |
| MA9 / C | +2.695［+2.663, +2.731］ | 18 |
| MA9 / A | -0.871［-0.909, -0.831］ | 18 |
| MA9 / Flux＋基础量化 | -2.887［-3.000, -2.817］ | 18 |
| CR3 / R | -4.105［-4.133, -4.076］ | 18 |
| CR3 / C | +0.566［+0.522, +0.612］ | 18 |
| CR3 / A | -3.078［-3.115, -3.033］ | 18 |
| CR3 / Flux＋基础量化 | -5.139［-5.226, -5.077］ | 18 |
| R / C | +4.487［+4.451, +4.521］ | 18 |
| R / A | +0.987［+0.954, +1.019］ | 18 |
| R / Flux＋基础量化 | -0.993［-1.095, -0.921］ | 18 |
| C / A | -3.664［-3.711, -3.618］ | 18 |
| C / Flux＋基础量化 | -5.737［-5.855, -5.647］ | 18 |
| A / Flux＋基础量化 | -1.999［-2.119, -1.916］ | 18 |

Experimental non-P2P NCCL staged transport v2; single-node 8×RTX 5090 TP8/DP1, fixed synthetic input. Local producer completion precedes packet exchange and local decode. Original cross-GPU tile arrival/compute overlap is not preserved. R/C/CR3 host submission and buffering now belong to the staged implementation, not the H100 transport. Arrival tables and selective masks remain frozen from H100; this is not a 5090 calibration. Quantization policies may have different numerical trajectories. Each trial is reported separately; all slow windows and negative gains are retained. Bootstrap intervals describe this fixed workload and are not adjusted for multiple comparisons.
