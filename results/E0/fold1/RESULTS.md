# ESL E0 单次固定划分结果

2026年11月6日完成。使用预先选定的已有 E1 第1折，训练186篇、验证25篇、测试47篇，共30轮。

| 实验 | P | R | 总体F1 | 句内F1 | 跨句F1 |
|---|---:|---:|---:|---:|---:|
| E0 BERT事件对分类器 | 60.12 | 59.41 | 59.76 | 71.55 | 55.12 |
| E1 CogECI Llama8B（同一折） | 71.32 | 53.53 | 61.16 | 79.69 | 51.29 |

E1 总体F1比 E0 高 1.39 个百分点。以上两行均为单次固定划分结果。

E0 验证集选中第25轮，验证F1为46.15。

## 计时与运行记录

- 30轮合计 1068.32 秒，约17分48.32秒。
- 每轮平均 35.61 秒。
- 计时覆盖训练、验证、验证改善时的测试及检查点保存。初始模型加载和短检查单独进行。
- 两张 RTX 3090，BERT句子编码并行，每步一篇文档。
- 最高温度为物理GPU0 69℃、GPU1 74℃，任务正常退出。
- 每轮耗时及累计耗时保存在 epoch_metrics.jsonl，完整30轮耗时在 results.json 的 wall_seconds。

## 实现与产物

E0 采用 BERT、事件跨度均值池化、[e1,e2,e1-e2] 和 MLP 2304→2304→2。特征组合和第1折由执行者预先选定，监督为事件对分类。
使用仓库 focal loss，alpha=0.75、gamma=2；seed=209，BERT学习率2e-5、分类器1e-4、dropout=0.4。
完整数据试跑发现的单句文档双卡空输入问题已修复，单句文档在主卡编码，多句文档由双卡分担。修复后经过包含单句及多句文档的训练、评估和模型重载检查。

- 最佳模型 `/home/dancer/tu/cogeci-reproduction/repo-priority/runs-e0/esl_fold1/fold_1_best.pt`，459,274,776字节。
- 预测 `/home/dancer/tu/cogeci-reproduction/repo-priority/runs-e0/esl_fold1/fold_1_predictions.json`。
- 划分 `/home/dancer/tu/cogeci-reproduction/repo-priority/runs-e0/esl_fold1/splits.json`。
- 代码 `/home/dancer/tu/cogeci-reproduction/repo-priority/scripts/train_e0.py`。
- 日志与温度记录 `/home/dancer/tu/cogeci-reproduction/repo-priority/runs-e0/esl_fold1/task.log`、`/home/dancer/tu/cogeci-reproduction/repo-priority/runs-e0/esl_fold1/temperature.csv`。

MAVEN 已按用户要求停止，已有检查点和日志保留。
