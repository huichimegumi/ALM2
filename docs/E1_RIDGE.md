# E1.4 lightweight scoring model: nested LOQO Ridge

E1.4 asks whether the frozen E1.2 representations and non-trained E1.3
criterion--chunk interactions contain learnable document-level scoring signal.
It intentionally uses only Ridge regression. Nonlinear heads and ranking losses
remain deferred so this experiment isolates the value of the representation.

## Leakage-safe protocol

- Outer split: leave one complete question out.
- Inner selection: up to five GroupKFold splits over the remaining questions.
- Selection objective: pooled within-question pairwise accuracy, then macro
  Spearman, then mean absolute error.
- Alpha grid: `0.01, 0.1, 1, 10, 100, 1000`.
- Four independent scoring heads, one per AEOLLM-2 dimension.
- Predictions are clipped to the official 0--10 range.
- Scaling, imputation, alpha selection, and fitting occur inside the outer fold.
- Official weighted-total pairwise accuracy is primary. Bootstrap uncertainty
  resamples the 10 questions and recomputes pooled correct/total pairs.

For the primary rubric models, each scoring head sees only its own dimension's
criterion-conditioned features. Criterion count, raw chunk count, and fixed
top-3/top-5 diagnostics are excluded.

## Ablations

- `global_{capped,unbounded}`: 1024-dimensional global document embedding.
- `rubric_{capped,unbounded}`: target-dimension primary cosine features.
- `structure`: E0 surface features, excluding generator/prompt metadata.
- `rubric_structure_{capped,unbounded}`: rubric plus structure.
- `all_{capped,unbounded}`: global plus rubric plus structure.
- `mean_loqo`: training-question mean baseline.

## Run

```bash
.venv-system-python-backup/bin/python scripts/run_e1_ridge.py \
  --output-dir outputs/e1/e1_4_accuracy
```

This stage is CPU-only. Outputs are written under `outputs/e1/e1_4/`, including
out-of-fold predictions, overall and per-question metrics, selected alphas,
question-bootstrap intervals, paired comparisons, the exact feature protocol,
and an automatically generated conclusions report.
