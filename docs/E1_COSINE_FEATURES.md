# E1.3 non-trained criterion--chunk interaction

E1.3 converts the frozen E1.2 vectors into fixed-schema, CPU-only features. It
does not use human scores and does not fit a model.

For every document and criterion it computes cosine similarity against every
chunk. The long audit table retains max, mean, standard deviation, quantiles,
top-fraction means, normalized log-mean-exp, fixed top-k diagnostics, and the
highest-scoring chunk identity.

Because held-out questions have different criteria, criterion positions cannot
be model columns. Per-criterion statistics are therefore aggregated within each
dimension using the official criterion weights. The fixed document table uses
weighted mean, minimum, and weighted standard deviation across criteria.

Global document features are the L2-normalized, structural-token-weighted mean
of chunk embeddings. Cache-derived length/type columns are diagnostics; E1.4
should still use the complete E0 surface-feature table for the structure-only
baseline.

## Run both representations

```bash
.venv-system-python-backup/bin/python scripts/run_e1_cosine_features.py \
  --cache-dir outputs/e1/embeddings/qwen3-0.6b-capped \
  --output-dir outputs/e1/features/qwen3-0.6b-capped

.venv-system-python-backup/bin/python scripts/run_e1_cosine_features.py \
  --cache-dir outputs/e1/embeddings/qwen3-0.6b-unbounded \
  --output-dir outputs/e1/features/qwen3-0.6b-unbounded
```

No GPU check is needed because this stage only loads NumPy arrays and computes
small matrix products on CPU.

Each output directory contains:

- `document_features.csv`: fixed-width model input without labels;
- `criterion_chunk_features.csv`: long-form audit and top-chunk evidence;
- `feature_manifest.json`: exact feature groups and definitions.

The primary rubric group uses count-normalized statistics. Fixed top-3/top-5
columns are explicitly marked diagnostic and should be excluded from the first
Ridge comparison.
