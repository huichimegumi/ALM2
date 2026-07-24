# AEOLLM-2 Rubric-conditioned Hierarchical Ranker

本仓库研究如何利用任务说明、评分 rubric 与长文档结构，对
AEOLLM-2 DeepResearch 报告进行多维评分和排序。当前实验从数据与指标审计
（E0）出发，依次验证冻结语义表示、criterion–chunk 交互、排序损失和最小
可学习交互。

当前统一采用 **accuracy-first** 协议：主要指标是官方 rubric 加权总分的
pairwise accuracy；Spearman 和 Kendall 用作辅助诊断。当前数据包含 10 个
question、160 个带标签文档和四个评分维度。

> 本仓库仍在持续实验中。根 README 只维护当前状态、复现入口和稳定约定；
> 每个实验的设计、消融和解释详见 [`docs/`](docs/)，精简结果详见
> [`results/`](results/)。

## 当前状态

| 阶段 | 实验目的 | 设备 | 状态 | 文档 | 当前结果 |
|---|---|---:|---|---|---|
| E0 | 数据、官方指标、表面/元数据混杂和历史 ODAT 审计 | CPU | 完成 | [E0](docs/E0_EXPERIMENT.md) | [结果](results/e0/e0_conclusions.md) |
| E1.1 | 保留 DOCX 结构的 capped/unbounded chunking | CPU | 完成 | [Chunking](docs/E1_CHUNKING.md) | [审计](docs/E1_UNBOUNDED_CHUNK_ANALYSIS.md) |
| E1.2 | 冻结 Qwen3 encoder，缓存 chunk/rubric embeddings | GPU | 完成 | [Embedding](docs/E1_EMBEDDING.md) | 缓存不纳入 Git |
| E1.3 | 构造非训练 cosine criterion–chunk 特征 | CPU | 完成 | [Cosine features](docs/E1_COSINE_FEATURES.md) | 特征不纳入 Git |
| E1.4 | Nested LOQO Ridge 表示消融 | CPU | 完成 | [Ridge](docs/E1_RIDGE.md) | [结果](results/e1/e1_4_accuracy/e1_4_conclusions.md) |
| E1.5 | Huber 与 Huber + pairwise loss 对照 | CPU | 完成 | [Pairwise MLP](docs/E1_PAIRWISE.md) | [结果](results/e1/e1_5_accuracy/e1_5_conclusions.md) |
| E1.6 | Rubric attribution、query 与 mismatch controls | GPU + CPU | 完成 | [Rubric controls](docs/E1_RUBRIC_CONTROLS.md) | [结果](results/e1/e1_6_accuracy/e1_6_conclusions.md) |
| E1.7 | Leakage-safe selective query Ridge | CPU | 完成，gate 未通过 | [Selective query](docs/E1_SELECTIVE_QUERY.md) | [结果](results/e1/e1_7_accuracy/e1_7_conclusions.md) |
| E2-A0 | Shared learned diagonal interaction | GPU | 完成，gate 未通过 | [E2-A0](docs/E2_A0_DIAGONAL_INTERACTION.md) | [结果](results/e2/e2_a0_accuracy/e2_a0_conclusions.md) |
| E2-A0.1 | Dimension-separated diagonal diagnostic | GPU | 完成，部分维度通过 | [E2-A0.1](docs/E2_A01_DIMENSION_SEPARATED.md) | [结果](results/e2/e2_a01_accuracy/e2_a01_conclusions.md) |

最新 accuracy-first 指标汇总见
[`results/accuracy_first_summary.md`](results/accuracy_first_summary.md)。
表中的最高 point estimate 可能来自 negative control，不应自动视为可用模型；
实验结论应以各阶段的预注册比较、paired bootstrap 和 gate 为准。

## 实验流程

```text
标签 CSV + DOCX 报告 + Rubric
              │
              ├── E0：数据、指标和基线审计
              │        └── surface_features.csv
              │
              └── E1.1：结构化切块
                       ├── capped chunks
                       └── unbounded chunks
                                │
                                ▼
                       E1.2：冻结 Embedding
                                │
                                ▼
                       E1.3：Cosine Features
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
             E1.4/E1.5      E1.6/E1.7       E2-A0/A0.1
             固定特征模型     Rubric controls   可学习交互
```

E1.2 生成的 embedding 会被后续阶段复用。E1.3 之后的 CPU 实验不需要重新
加载 encoder；E1.6 只额外编码少量 query variants；E2 直接复用已有缓存。

## 仓库结构

```text
.
├── scripts/                 # 实验命令入口与报告刷新脚本
├── src/                     # 数据、特征、模型和统计实现
│   ├── aeollm_e0/
│   └── aeollm_e1/
├── tests/                   # 单元测试
├── docs/                    # 各阶段实验设计和解释
├── results/                 # Git 跟踪的精简结果
├── outputs/                 # 本地完整运行产物，不纳入 Git
├── data/                    # 标签、rubric 与本地报告数据
├── legacy/                  # 历史代码、映射和预测
└── requirements*.txt        # 最小依赖与环境快照
```

目录约定：

- `outputs/` 保存预测、checkpoint、embedding、特征和完整运行报告，默认被
  `.gitignore` 排除。
- `results/` 只保存适合版本控制的指标、bootstrap、超参数选择和结论。
- 无后缀目录（如 `outputs/e1/e1_4/`）是历史协议产物；当前结果统一使用
  `*_accuracy` 目录。不要用新代码覆盖历史目录。
- `protocol.yaml` 是判断一次运行采用何种输入、路径和选择目标的首要依据；
  `run_status.json` 用于确认运行是否完整结束。

## 环境

所有命令均应从仓库根目录执行。脚本会自行把 `src/` 加入 Python path，
无需安装本地 package。

### CPU 实验环境

E0、chunking、cosine features 和 Ridge 实验的最小环境：

```bash
python3 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-e0.txt
```

`requirements-e0.txt` 包含 NumPy、pandas、SciPy、scikit-learn、
python-docx、PyYAML、tabulate 和 pytest。

### 完整 GPU 环境

Embedding 和 E2 还需要与本机 CUDA 匹配的 PyTorch、Transformers 及相关
依赖。`requirements.lock.txt` 是当前实验服务器的完整环境快照，其中包含
CUDA 13、PyTorch、Transformers 和 vLLM 等固定版本；它适合复现当前服务器
环境，但不保证能直接安装在不同 CUDA/驱动组合的机器上。

安装 GPU 环境前应先根据目标服务器选择兼容的 PyTorch/CUDA 版本，再安装
其余依赖。可用以下命令记录环境：

```bash
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

## 数据准备

默认命令期望以下结构：

```text
data/
├── official/hf-aeollm/aeollm-2-train/
│   ├── train_deepresearch.csv
│   └── rubric_dataset/
└── incoming/google-drive/train/
    └── Report*/Doc_*.docx

legacy/aeollm2_train_code/
├── prompts/mapping_key_Readability.xlsx
└── outputs/
```

其中正式数据、DOCX 报告和大体积中间产物不会提交到 Git。各脚本均提供路径
参数；如果数据不在默认位置，请使用 `--labels`、`--report-root`、
`--rubric-dir` 等参数覆盖。

运行实验前建议先检查：

```bash
test -f data/official/hf-aeollm/aeollm-2-train/train_deepresearch.csv
test -d data/official/hf-aeollm/aeollm-2-train/rubric_dataset
test -d data/incoming/google-drive/train
```

## Quick start

激活包含项目依赖的环境后：

```bash
# 查看参数
python scripts/run_e0.py --help

# 运行全部单元测试
python -m pytest -q

# CPU-only E0 smoke run；缩短 bootstrap，仅用于检查流程
python scripts/run_e0.py \
  --bootstrap-resamples 100 \
  --output-dir outputs/e0_smoke
```

完整 E0 accuracy-first 运行：

```bash
python scripts/run_e0.py \
  --output-dir outputs/e0_accuracy
```

## 完整复现

以下命令按依赖顺序排列。`cuda:N` 中的 `N` 必须替换为运行时实际空闲的
GPU index。

### 1. E0：审计和 surface features

```bash
python scripts/run_e0.py \
  --output-dir outputs/e0_accuracy
```

主要产物包括 `surface_features.csv`、out-of-fold predictions、
`model_metrics.csv`、`paired_bootstrap.csv`、`protocol.yaml` 和
`e0_conclusions.md`。

### 2. E1.1：构造 capped 与 unbounded chunks

```bash
# 默认最多 96 chunks/document
python scripts/run_e1_chunking.py \
  --output-dir outputs/e1

# 保留自然 chunk 数量，不做 document-level cap
python scripts/run_e1_chunking.py \
  --max-chunks 0 \
  --output-dir outputs/e1_unbounded
```

该阶段确定性、CPU-only，不执行检索、评分或模型推理。

### 3. E1.2：生成两套冻结 embedding cache

每次 GPU 运行前都应重新检查设备：

```bash
nvidia-smi
```

然后显式指定空闲设备：

```bash
# Capped
python scripts/run_e1_embedding.py \
  --input outputs/e1/chunked_documents.jsonl \
  --output-dir outputs/e1/embeddings/qwen3-0.6b-capped \
  --device cuda:N \
  --batch-size 8

# 再次执行 nvidia-smi 后运行 unbounded
python scripts/run_e1_embedding.py \
  --input outputs/e1_unbounded/chunked_documents.jsonl \
  --output-dir outputs/e1/embeddings/qwen3-0.6b-unbounded \
  --device cuda:N \
  --batch-size 8
```

脚本拒绝不带 index 的 `cuda`，并在分配模型前检查所选 GPU 是否空闲。输入
超过 `--max-length` 时会报错，不会静默截断。只有确认需要重建缓存时才使用
`--overwrite`。

### 4. E1.3：生成 cosine interaction features

```bash
python scripts/run_e1_cosine_features.py \
  --cache-dir outputs/e1/embeddings/qwen3-0.6b-capped \
  --output-dir outputs/e1/features/qwen3-0.6b-capped

python scripts/run_e1_cosine_features.py \
  --cache-dir outputs/e1/embeddings/qwen3-0.6b-unbounded \
  --output-dir outputs/e1/features/qwen3-0.6b-unbounded
```

该阶段 CPU-only，输出固定宽度的 document features 和长表形式的
criterion–chunk 审计特征。

### 5. E1.4：Nested LOQO Ridge

```bash
python scripts/run_e1_ridge.py \
  --surface-features outputs/e0_accuracy/surface_features.csv \
  --output-dir outputs/e1/e1_4_accuracy
```

内层 grouped validation 依次最大化 pairwise accuracy、Spearman，并以 MAE
作为最后的数值 tie-breaker。

### 6. E1.5：固定 MLP 与 pairwise loss

```bash
python scripts/run_e1_pairwise.py \
  --surface-features outputs/e0_accuracy/surface_features.csv \
  --ridge-reference outputs/e1/e1_4_accuracy/predictions/all_unbounded.tsv \
  --output-dir outputs/e1/e1_5_accuracy
```

E1.5 的 architecture、epoch 和 seed ensemble 固定，没有依据 held-out
question 做 early stopping 或模型选择。这里的 pairwise loss 是训练消融；
它与“用 pairwise accuracy 选择 Ridge 超参数”是两个不同概念。该阶段虽然
只在 CPU 上运行，但仍需要 PyTorch。

### 7. E1.6：query variants 与 rubric controls

先只编码 query variants：

```bash
nvidia-smi

python scripts/run_e1_query_variants.py \
  --input outputs/e1_unbounded/chunked_documents.jsonl \
  --output-root outputs/e1/embeddings/qwen3-0.6b-query-variants \
  --device cuda:N \
  --max-length 4096 \
  --batch-size 8
```

再在 CPU 上构造 controls 并运行 nested LOQO Ridge：

```bash
python scripts/run_e1_6.py \
  --surface-features outputs/e0_accuracy/surface_features.csv \
  --output-dir outputs/e1/e1_6_accuracy
```

### 8. E1.7：selective query Ridge

```bash
python scripts/run_e1_7.py \
  --surface-features outputs/e0_accuracy/surface_features.csv \
  --criterion-only-features \
    outputs/e1/e1_6_accuracy/control_features/criterion_only \
  --output-dir outputs/e1/e1_7_accuracy
```

该阶段复用 matched 与 criterion-only 特征，不需要重新生成 embedding。

### 9. E2-A0：shared diagonal interaction

```bash
nvidia-smi

python scripts/run_e2_a0.py \
  --device cuda:N \
  --surface-features outputs/e0_accuracy/surface_features.csv \
  --output-dir outputs/e2/e2_a0_accuracy
```

E2 checkpoint 会记录协议并支持按 outer fold 恢复。只有在有意改变固定协议并
确认旧 checkpoint 不再适用时，才使用 `--overwrite-checkpoints`。

### 10. E2-A0.1：dimension-separated diagnostic

```bash
nvidia-smi

python scripts/run_e2_a01.py \
  --device cuda:N \
  --surface-features outputs/e0_accuracy/surface_features.csv \
  --shared-a0-predictions \
    outputs/e2/e2_a0_accuracy/predictions/diagonal_matched_hybrid.tsv \
  --output-dir outputs/e2/e2_a01_accuracy
```

### 11. 刷新 accuracy-first 统计和报告

```bash
python scripts/refresh_accuracy_first_reports.py
```

该脚本读取已经存在的 `*_accuracy` out-of-fold predictions，重新生成
question-level bootstrap、paired comparisons、各阶段结论和
`outputs/accuracy_first_summary.md`。它不会重新训练模型，也不应被用于历史
无后缀目录。

## 脚本索引

| 脚本 | 阶段 | 主要输入 | 主要输出 | 设备 |
|---|---|---|---|---|
| `scripts/run_e0.py` | E0 | 标签、DOCX、rubric、legacy predictions | 审计、surface features、OOF predictions | CPU |
| `scripts/run_e1_chunking.py` | E1.1 | DOCX、标签、rubric | chunked JSONL 与审计表 | CPU |
| `scripts/run_e1_embedding.py` | E1.2 | chunked JSONL | embedding cache | GPU/CPU |
| `scripts/run_e1_cosine_features.py` | E1.3 | embedding cache | document/criterion features | CPU |
| `scripts/run_e1_ridge.py` | E1.4 | E0 与 E1.3 features | Ridge OOF predictions | CPU |
| `scripts/run_e1_pairwise.py` | E1.5 | features、Ridge reference | MLP OOF predictions | CPU |
| `scripts/run_e1_query_variants.py` | E1.6 前置 | unbounded chunks | query embedding caches | GPU/CPU |
| `scripts/run_e1_6.py` | E1.6 | base/query caches、features | controls 与 Ridge results | CPU |
| `scripts/run_e1_7.py` | E1.7 | matched/control features | selective query results | CPU |
| `scripts/run_e2_a0.py` | E2-A0 | embeddings、features | diagonal checkpoints/results | GPU/CPU |
| `scripts/run_e2_a01.py` | E2-A0.1 | E2-A0 与共享缓存 | separated checkpoints/results | GPU/CPU |
| `scripts/refresh_accuracy_first_reports.py` | 报告 | 已有 OOF predictions | bootstrap、报告与汇总 | CPU |

完整参数始终以代码为准：

```bash
python scripts/<script>.py --help
```

E2 的入口会先检查显式 device，因此查看这两个脚本的帮助时也要提供 device：

```bash
python scripts/run_e2_a0.py --device cpu --help
python scripts/run_e2_a01.py --device cpu --help
```

## 评估协议

当前统一协议详见
[`docs/ACCURACY_FIRST_PROTOCOL.md`](docs/ACCURACY_FIRST_PROTOCOL.md)：

- Outer evaluation：Leave-One-Question-Out，共 10 个 outer folds。
- Grouped inner selection：验证 question 保持完整。
- Primary metric：官方 rubric-weighted total pairwise accuracy。
- Tie-breakers：macro within-question Spearman，然后 document-level MAE。
- Secondary metrics：Spearman、Kendall 和 dimension-level diagnostics。
- Bootstrap unit：question，而不是 document pair。
- Held-out question 不参与 imputation、scaling、超参数/query 选择、拟合或
  early stopping。

Pairwise accuracy 是主要选择和报告指标，但并不意味着所有模型都以 pairwise
loss 训练。Ridge 仍拟合连续分数；E1.5 显式比较 Huber 和 pairwise loss；
E2 保留各自预先固定的训练目标。

## 输出说明

每个完整实验通常包含：

| 文件 | 含义 |
|---|---|
| `protocol.yaml` | 输入路径与 hash、seed、环境、split 和选择协议 |
| `run_status.json` | 是否完成、模型/文档数量、设备与耗时 |
| `predictions/*.tsv` | Outer LOQO out-of-fold predictions |
| `model_metrics.csv` | 总体 accuracy、Spearman、Kendall、MAE 等 |
| `per_question_metrics.csv` | 每个 question 的指标和正确 pair 数 |
| `bootstrap_ci.csv` | question-bootstrap 置信区间 |
| `paired_bootstrap.csv` | 候选与 reference 的 paired comparisons |
| `selected_hyperparameters.csv` | 每个 outer fold 的内层选择 |
| `training_diagnostics.csv` | 训练型实验的 fold/seed 诊断 |
| `*_conclusions.md` | 自动生成的主要发现、gate 和解释 |
| `checkpoints/` | 可恢复的 GPU/长时间实验中间状态 |

确认一次运行可用于分析前，至少检查：

```bash
cat outputs/<experiment>/run_status.json
sed -n '1,120p' outputs/<experiment>/protocol.yaml
```

`status: complete` 只能说明流程完成；是否支持研究假设仍应查看 paired
comparison、置信区间和实验 gate。

## 测试

```bash
python -m pytest -q
```

测试覆盖数据完整性、官方 tie semantics、指标一致性、DOCX 解析、chunking、
embedding cache、cosine features、nested LOQO、pairwise training 和 E2
interaction/checkpoint 行为。

## 常见问题

### `ModuleNotFoundError`

当前 shell 没有激活包含依赖的环境。先激活 CPU 或 GPU virtualenv，再运行
脚本。不要假设系统 `python3` 已安装 NumPy、python-docx 或 PyTorch。

### GPU runner 拒绝启动

Embedding/E2 runner 要求 `--device cuda:N`，并会在分配张量前检查该卡的
显存占用和 utilization。若设备忙碌，请重新选择空闲卡或等待；不要绕过检查
与其他实验共享已占用 GPU。

### 已存在的缓存或 checkpoint 与当前协议不一致

优先使用新的输出目录。`--overwrite` 和 `--overwrite-checkpoints` 会改变
已有运行状态，只应在确认输入或协议有意变化时使用。

### 为什么同时存在历史目录与 `*_accuracy` 目录

历史实验曾使用 MAE 等作为内层选择目标。accuracy-first 更新后使用新目录
重新运行，以保留旧预测并避免把回顾性分析误认为新的 confirmatory result。
当前文档和精简结果只以 `*_accuracy` 为准。

### `outputs/` 中有结果，但 Git 看不到

这是预期行为。完整产物通常很大并被 `.gitignore` 排除。需要发布结果时，只
同步可审计的精简文件到 `results/`，不要提交 embedding、全部 checkpoints
或重复缓存。

## 继续实验时如何维护

新增或重跑实验时遵循以下约定：

1. 为新实验使用唯一名称和新输出目录，不覆盖历史 predictions。
2. 在 `scripts/` 提供单一入口，并使全部输入/输出路径可通过 CLI 覆盖。
3. 在 `protocol.yaml` 记录数据 hash、seed、split、选择目标、环境和依赖路径。
4. 在 `run_status.json` 记录 completion、数据规模、设备和必要诊断。
5. 在 `docs/` 新建或更新实验设计，写清假设、controls、gate 和解释边界。
6. 将精简指标和结论同步到 `results/`，不要提交大体积缓存。
7. 更新本 README 的“当前状态”和实验流程；详细消融仍留在阶段文档中。
8. 添加或更新测试，并运行 `python -m pytest -q`。

建议的新实验完成检查表：

```text
[ ] 脚本可通过 --help 查看参数
[ ] 默认路径不引用历史实验目录
[ ] CPU/GPU 要求明确
[ ] outer held-out question 未参与选择或拟合
[ ] protocol.yaml 与 run_status.json 完整
[ ] paired bootstrap 以 question 为重采样单位
[ ] conclusions 明确区分 point estimate、negative control 与正式 gate
[ ] results/ 已同步必要的精简产物
[ ] README 状态表和 docs 索引已更新
[ ] 全部测试通过
```
