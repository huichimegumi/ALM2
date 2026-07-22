# E1.2 frozen encoder

## Question answered by this stage

E1.2 only asks whether a fixed multilingual semantic space contains useful
criterion--chunk alignment signal. It does not train a scoring head and does not
ask an LLM to assign scores.

## Controlled first encoder

- Primary encoder: `Qwen/Qwen3-Embedding-0.6B`.
- Chunk/document side: the exact E1.1 chunk text, without an instruction.
- Criterion/query side: original task + criterion name + explanation, preceded
  by one shared English retrieval instruction.
- Criterion weights are retained in the index but are not embedded in the text.
- Pooling: last non-padding token; all vectors are L2-normalized.
- Model state: evaluation mode, inference mode, and every parameter frozen.
- Cache dtype: float32 for the first experiment. Float16 is a storage ablation,
  not the default.

The same instruction is used for every dimension. This avoids reintroducing the
dimension-specific prompt engineering that E1 is intended to replace.

## Two required caches

The 96-chunk dataset and the unbounded structural dataset must both be embedded.
They contain 11,385 and 14,841 chunks respectively; each contains the same 210
question-level criteria.

```bash
# On tyan: inspect all GPUs and choose an actually idle index first.
nvidia-smi

# Capped structural input.
.venv-system-python-backup/bin/python scripts/run_e1_embedding.py \
  --input outputs/e1/chunked_documents.jsonl \
  --output-dir outputs/e1/embeddings/qwen3-0.6b-capped \
  --device cuda:ID

# Recheck GPU state before starting the second run.
nvidia-smi

# Unbounded structural input.
.venv-system-python-backup/bin/python scripts/run_e1_embedding.py \
  --input outputs/e1_unbounded/chunked_documents.jsonl \
  --output-dir outputs/e1/embeddings/qwen3-0.6b-unbounded \
  --device cuda:ID
```

The runner rejects bare `cuda`, queries the selected card immediately before
model allocation, and refuses it when used memory or utilization exceeds the
configured idle thresholds. It also aborts if any input would be silently
truncated. The default Qwen-tokenizer limit is 4096. The observed capped-data
maximum is 3557 Qwen tokens, despite E1.1's 512-token structural limit, because
the two token definitions are different. The default batch size is reduced to
8 to keep the longest batches conservative.

## Required cache audit

Before E1.3, verify both manifests:

- 160 documents, 10 questions, and 210 criteria;
- 11,385 capped or 14,841 unbounded chunks;
- zero truncated inputs;
- finite, normalized 1024-dimensional vectors;
- a resolved model revision and the correct input SHA-256;
- the expected explicit GPU index and its idle-check observation.

Do not select between capped and unbounded representations using test/leaderboard
results. E1.3 must compare them under the same leave-one-question-out folds.

## Deferred ablations

Run `BAAI/bge-m3` only as a second-encoder ablation after the complete Qwen cache
and E1.3 Ridge result exist. Do not test many embedding models at once: with only
10 independent questions, model shopping can overfit the validation protocol.

The first E1.3 comparison should be:

1. global document representation only;
2. criterion--chunk cosine statistics only;
3. E0 structure features only;
4. all three groups;
5. capped versus unbounded input.

Only if criterion-conditioned features improve over the global representation
consistently across questions should the project proceed to learned MIL pooling.
