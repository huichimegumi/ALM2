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
