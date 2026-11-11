# 第十组：CogECI 文档级事件因果关系识别


## 结果与文件

三组均已完成 seed=209 的同划分五折。先逐折计算指标，再取算术平均；分数单位为 %。

| 实验 | 改动 | P | R | F1 | 句内 F1 | 跨句 F1 | 五折训练阶段/h | 一次生成/h |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | BERT 事件对分类基线 | 62.32 | 64.78 | 63.48 | 72.93 | 59.70 | 1.50 | 0 |
| E1 | Llama-3-8B 伪标签 + CogECI 修正版 | 69.47 | 60.85 | 64.73 | 77.58 | 58.12 | 11.60 | 1.61 |
| E2 | 只将伪标签模型换成 Qwen2.5-7B | 69.28 | 59.24 | 63.81 | 77.36 | 56.66 | 11.87 | 1.49 |

E0 使用两张 RTX 3090 做句子编码，E1/E2 正式 ESL 运行使用单卡。计时包含验证、测试和保存等阶段，边界也不同，不能解释为受控的速度倍率比较。

- `code/`：模型、训练、伪标签生成和检查入口。
- `configs/`：正式运行参数、模型 revision 与运行记录。
- `data_info/`：来源、哈希、各折统计和检查记录，不含完整输入缓存。
- `models/`：GitHub Release 权重地址、版本记录和 SHA256，运行下载脚本后保存 E0/E1/E2 第 1 折最佳权重。
- `prompts/`：两个本地模型的完整提示、原始回答、解析失败记录及训练用关键句 CSV。
- `results/E0/`、`E1/`、`E2/`：五折日志、选择轮次与逐对预测（JSON/CSV）。
- `results/comparison.csv`：同时保留第 1 折和完整五折，原始分数取值为 0–1，时间单位为秒。
- `provenance/`：修改记录、实际环境清单、早期冻结代码与运行检查。

E0 第 2 折使用完整完成的 `esl_fold2_resume_20261106`；中断和 smoke 记录只作为历史保留，不参与五折平均。

## 1. 先检查归档结果（仅需 Python 标准库、CPU）

在解压后的包根目录执行：

```bash
python code/check_submission.py
```

它核对 233 篇文档恰好覆盖测试集合、各集合无交集、15 折测试 P/R/F1 与句内/跨句 F1，并检查 7 个报告案例的金标和预测。通过时输出 `validated_completed_folds: 15`。

## 2. 准备训练环境与数据

训练入口面向 Linux/CUDA；温度监控使用 Linux 进程组。正式环境为 Python 3.10.22、PyTorch 2.1.0+cu118、Transformers 4.36.2、torch-geometric 2.3.1、torch-scatter 2.1.2+pt21cu118。完整实际环境见 `provenance/training_freeze.txt`。上游模型目录中的旧 `requirements.txt` 是原仓库记录，不代表本次实际版本。

一个对应的环境准备顺序如下（PyTorch 与 PyG 扩展需要匹配 CUDA）：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter==2.1.2+pt21cu118 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install -r code/requirements-training.txt
python code/prepare_cache.py
python -c "from transformers import BertModel,BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"
```

`prepare_cache.py` 只下载固定 iLIF 提交中的两个 ESL 缓存，并验证 SHA256。也可从原项目拷贝这两个文件到 `data_info/cache/`，再运行同一检查。来源和划分见 `data_info/README.md`。模型代码读取的原始标签不作修改。

## 3. 重载已训练模型，预测一个真实文档

第 1 折最佳权重发布在[GitHub Release](https://github.com/TT-t-1/CogECI-ESL/releases/tag/v1.0.0)。三个权重的下载入口如下。

Release 中提供三个权重的二进制分片。下载脚本会按顺序下载、自动合并到对应模型目录，并核验完整文件的 SHA256。

先在解压后的包根目录执行以下命令。脚本读取 `models/github_release.json` 中记录的 Release 地址，将三个权重保存到 `models/E0/`、`models/E1/`、`models/E2/`，并核验 `models/checksums.json` 的 SHA256。

```bash
python code/download_weights.py
```

随后运行下面命令，加载第 1 折最佳权重，预测测试文档 `1_1`，并逐对比较归档预测。E1/E2 每次一张卡，E0 原实现要求两张可见卡。

```bash
CUDA_VISIBLE_DEVICES=1 python code/demo.py --experiment E1 --output E1_demo.json
CUDA_VISIBLE_DEVICES=1 python code/demo.py --experiment E2 --output E2_demo.json
CUDA_VISIBLE_DEVICES=0,1 python code/demo.py --experiment E0 --output E0_demo.json
```

成功时 `checkpoint_reload_matches_archive` 为 `true`。本次已在原服务器用相同入口验证 E0/E1/E2，文档 `1_1` 的 496 个事件对均与归档预测完全一致，见 `provenance/submission_validation.json`。此入口不重新训练。原权重和归档预测均按原始运行保存；运行环境差异可能导致预测变化，脚本会直接报出不一致。

## 4. 重新训练

输出必须使用新的空目录，避免覆盖归档结果。包装入口只适配路径，不改训练数学逻辑。E0 每次指定一折；E1/E2 的原 `main.py` 一次运行完整五折，`--fold` 只对 E0 生效。

```bash
python code/reproduce.py E0 --fold 1 --gpus 0,1 --run-dir reruns/E0_fold1
python code/reproduce.py E0 --fold 2 --gpus 0,1 --run-dir reruns/E0_fold2
python code/reproduce.py E0 --fold 3 --gpus 0,1 --run-dir reruns/E0_fold3
python code/reproduce.py E0 --fold 4 --gpus 0,1 --run-dir reruns/E0_fold4
python code/reproduce.py E0 --fold 5 --gpus 0,1 --run-dir reruns/E0_fold5
python code/reproduce.py E1 --gpus 1 --run-dir reruns/E1
python code/reproduce.py E2 --gpus 1 --run-dir reruns/E2
```

默认使用包内已经生成的标签，复现分类训练不需要载入 7B/8B 模型。监控器记录 GPU 温度，达到 85℃ 时停止自己启动的任务。`--epoch 1` 可用于检查运行流程，所得结果不属于报告的 30 轮正式实验。

## 5. 可选：重新生成伪标签

生成使用独立环境（Transformers 4.45.2、accelerate 0.34.2、同版 PyTorch）；见 `provenance/llm_freeze.txt`。自行准备已获访问权限的 Meta-Llama-3-8B-Instruct 和 Qwen2.5-7B-Instruct 本地权重，revision 见 `configs/runtime_manifest.json`。下面的本地权重路径需替换为实际目录。

```bash
COGECI_GPUS=1 python code/scripts/guard.py reruns/label_llama python code/scripts/label_local.py --model /path/to/Meta-Llama-3-8B-Instruct --name llama8b --dataset esl --output_dir reruns/label_llama/labels
COGECI_GPUS=1 python code/scripts/guard.py reruns/label_qwen python code/scripts/label_local.py --model /path/to/Qwen2.5-7B-Instruct --name qwen7b --dataset esl --output_dir reruns/label_qwen/labels
```

新生成目录不会覆盖归档标签。要用新标签训练，可在原模型目录运行 `main.py`，设置 `COGECI_DATA_DIR` 为新标签目录，并沿用 `reproduce.py` 所列其余参数。完整提示在源代码 `llm/run_llm_processing.py`、`label_local.py` 和原始 `responses.jsonl` 中。

