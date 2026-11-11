# ESL 数据来源与实际划分

本包不包含完整输入缓存。原始语料来源：<https://github.com/cltl/EventStoryLine/>。

本次实际使用 iLIF 仓库 <https://github.com/LchengC/iLIF> 固定提交 `d89b0267051183da52d59ac5bb8d2b3fa775c8eb` 的 `data/database/exist/` 下两个文件：

| 文件 | SHA256 |
|---|---|
| EventStoryLine_bert_intra_and_inter_train_features.pkl | 728c00c43f74c34a95883984c02e7a78d553be5186f9e83df8c954ed71a18999 |
| EventStoryLine_bert_intra_and_inter_dev_features.pkl | c128c11485793ba8095d1cf2b56abb665cb57c7cc60c0a0305b1672940eb057f |

包根目录执行 `python code/prepare_cache.py`，下载至 `data_info/cache/` 并校验。缓存需要原仓库的 Python 类定义来反序列化，载入函数位于 `code/scripts/prepare_data.py`。

文件名中的 train 是 233 篇训练/测试轮换池，并非每折的最终训练集合；dev 为固定 25 篇验证文档。五折实际文档 ID 见 `results/E1/splits.json`。E0/E1/E2 使用相同 ID，seed=209；每折训练 186/187 篇，测试 47/46 篇。五折后 233 篇各作为一次测试文档。

`split_audit.json` 列出每折各集合的文档、事件、候选对和正例统计，以及交集、重复键、冲突标签检查。稳定预测样本 ID 格式为 `doc_id/pair/pair_index`，对应缓存中 `event_pairs`、`target` 的相同位置。`rel_type=0/1` 分别为句内/跨句；`target=0/1` 为无标注关系/有标注关系。当前缓存不编码因果方向。

`audit.json` 是原运行的多数据源审计记录，当前提交仅采用其 ESL 部分。`esl_source_audit.json` 另行记录缓存与原始语料映射检查中的不完全一致，不能把缓存审计结果解释为与 XML 全量逐项完全等同。实验继续使用固定缓存金标，未据此改标签。
