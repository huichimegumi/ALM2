# E1.1 structure-preserving chunking

E1.1 creates one deterministic JSONL input shared by all later frozen-embedding
experiments. It performs no retrieval, scoring, model inference, or GPU work.

## Rules

- Top-level DOCX paragraphs and tables retain their original Word order.
- Explicit Word heading styles and outline levels are headings. A conservative
  numbered/named-heading rule recovers headings incorrectly styled as `Normal`.
- Headings remain standalone chunks during normal chunking.
- Adjacent short paragraphs or adjacent list items of the same type are merged;
  merging never crosses a heading or table boundary.
- Body blocks longer than 512 simple multilingual tokens are split into balanced
  contiguous slices. The tokenizer counts English words/numbers, individual CJK
  characters, URLs, and punctuation; the exact embedding tokenizer is deliberately
  deferred until E1.2 is selected.
- Tables are serialized row-by-row and split only at row boundaries when possible.
- The default cap is 96 chunks. If normal structural chunking exceeds the cap,
  the smallest adjacent chunks are compacted while preserving every source block
  and its order. No top-k selection or content dropping is used. Such chunks and
  documents are explicitly flagged for audit.

The cap can be disabled to audit natural within-question length variation:

```bash
.venv-system-python-backup/bin/python scripts/run_e1_chunking.py \
  --max-chunks 0 \
  --output-dir outputs/e1_unbounded
```

Every chunk records source block IDs/types, split metadata, token count, and whether
the 96-chunk budget forced an overflow merge. Per-block metadata records Word style
and the reason a paragraph was classified as a heading, making the heuristic
auditable without duplicating all source text.

## Run

```bash
.venv-system-python-backup/bin/python scripts/run_e1_chunking.py
```

Outputs are written to `outputs/e1/`:

- `chunked_documents.jsonl`: prompt, rubric, labels, hashes, and ordered chunks.
- `chunking_summary.csv`: per-document block/chunk statistics and audit flags.
- `per_question_chunking_summary.csv`: within-question chunk dispersion and its
  relationship with document token count.
- `chunking_run.json`: run-level configuration and counts.

This stage is CPU-only. E1.2 must recheck GPU availability immediately before
embedding generation and bind only an idle device.

## Next stage: E1.2 frozen encoder

E1.2 uses `Qwen/Qwen3-Embedding-0.6B` as the first controlled encoder. Each
criterion query contains the original task, criterion name, and explanation.
All four dimensions share one neutral English retrieval instruction; report
chunks are encoded without an instruction. The encoder is put in evaluation
mode, all parameters are frozen, and only normalized vectors are cached.

The cache separates immutable arrays from auditable JSONL indices:

```text
outputs/e1/embeddings/qwen3-0.6b/
├── chunk_embeddings.npy
├── criterion_embeddings.npy
├── chunk_index.jsonl
├── criterion_index.jsonl
└── embedding_manifest.json
```

On the tyan server, inspect GPU state first, choose an idle card, and pass its
index explicitly. The script performs a second `nvidia-smi` check immediately
before allocating the model and refuses a busy card. Bare `cuda` is rejected.

```bash
nvidia-smi
.venv-system-python-backup/bin/python scripts/run_e1_embedding.py \
  --device cuda:ID \
  --batch-size 8
```

Inputs longer than `--max-length` cause a hard error instead of silent
truncation. The 4096-token default accommodates the observed maximum of 3557
Qwen tokens in the capped training chunks; E1.1's simple multilingual token
counter and the Qwen tokenizer are not equivalent. Increase it only if the
preflight reports an offending chunk or criterion. The resulting manifest pins the resolved model
revision, source JSONL hash, token-length maxima, device, and GPU idle-check
observation. E1.3 should consume this cache without loading the encoder again.
The complete two-cache protocol and acceptance checks are in
[`E1_EMBEDDING.md`](E1_EMBEDDING.md).
