# E2-A0 minimal learned diagonal criterion--chunk interaction

E2-A0 tests the smallest learnable replacement for E1's fixed cosine
interaction. The Qwen3-Embedding-0.6B encoder remains frozen, all unbounded
chunks are retained, criterion weights remain fixed to the official rubric, and
the same four pooling operators are used by both interaction models.

## Models

The fixed control uses:

```text
s_ij = cosine(r_i, h_j)
```

The learned model adds one shared 1024-dimensional diagonal metric:

```text
s_ij = cosine(r_i, h_j)
     + sqrt(1024) * dot(r_i * w, h_j)
```

`w` is initialized to zero, so the learned interaction begins exactly at the
fixed-cosine control. Criterion--chunk scores are reduced with mean, maximum,
top-10%-mean, and normalized log-mean-exp pooling. Criterion evidence is then
aggregated with the official criterion weights.

Only comprehensiveness and instruction following receive interaction residuals.
Insight and readability remain the predictions of an outer-fold
`global embedding + structure` Ridge model. This follows E1.6, which found
criterion-specific gains for the two alignment dimensions but no aggregate
benefit from applying the rubric branch uniformly to all four heads.

## Leakage-safe protocol

- Outer evaluation is leave-one-question-out.
- The no-rubric Ridge offset is refitted inside every outer fold.
- Ridge alpha selection uses grouped inner cross-validation.
- Target standardization uses outer-training documents only.
- Epoch count, optimizer settings, pooling, and architecture are fixed before evaluation.
- Predictions are averaged over three fixed seeds.
- The held-out question is never used for training or early stopping.
- Bootstrap uncertainty resamples the ten questions.

The formal controls are:

```text
global + structure Ridge
matched fixed cosine
matched learned diagonal
generic-dimension learned diagonal
five mismatched-rubric learned diagonals
five-mismatch prediction ensemble
```

The primary gate compares matched learned diagonal against matched fixed cosine
on the question-macro mean of comprehensiveness and instruction-following
Spearman. Passing requires a positive mean delta, bootstrap `P(delta > 0) >= 0.90`,
and at least seven of ten held-out questions improving.

## Run

Inspect every GPU and choose an idle device:

```bash
nvidia-smi
```

Then run with the explicit device index:

```bash
.venv-system-python-backup/bin/python scripts/run_e2_a0.py \
  --device cuda:N
```

The CLI checks the selected GPU again immediately before loading tensors. It
rejects a busy GPU instead of silently allocating on it. Fold-level baseline and
interaction checkpoints are written under `outputs/e2/e2_a0/checkpoints/`, so an
interrupted run resumes without repeating completed folds. Use
`--overwrite-checkpoints` only when intentionally changing the fixed protocol.

Outputs include full out-of-fold predictions, overall and per-question metrics,
alignment-specific paired bootstraps, baseline alpha selections, training
diagnostics, exact mismatch maps, protocol metadata, and an automatic decision
report.
