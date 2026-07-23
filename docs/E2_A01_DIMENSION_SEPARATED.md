# E2-A0.1 dimension-separated diagonal diagnostic

E2-A0.1 tests whether E2-A0 failed because one diagonal metric was shared by
comprehensiveness and instruction following. It changes only that architectural
assumption. Each dimension receives an independent 1024-dimensional diagonal
while the existing dimension-specific four-pooling heads remain unchanged.

The frozen encoder, unbounded chunks, full matched queries, official criterion
weights, four pooling operators, global-plus-structure Ridge offset, Huber
objective, joint two-head training, optimizer, epochs, three seeds, and outer
LOQO splits remain fixed. The fixed-cosine control uses the same joint training
as E2-A0, giving the learned model a like-for-like control.

The gate is applied separately to each dimension. Passing requires:

- positive mean question-level Spearman delta over its fixed-cosine head;
- bootstrap `P(delta > 0) >= 0.90`;
- at least 7 of 10 held-out questions improving;
- no mean Spearman loss relative to global + structure.

Generic and five cyclic mismatched-rubric controls are trained only for a
dimension that passes. The existing shared E2-A0 out-of-fold predictions are
loaded as an additional diagnostic when available. Criterion-only
comprehensiveness is intentionally excluded; E2-A0.1b should be attempted only
if the comprehensiveness branch passes this diagnostic.

## Run

Inspect all GPUs immediately before the experiment:

```bash
nvidia-smi
```

Then select an idle GPU:

```bash
.venv-system-python-backup/bin/python scripts/run_e2_a01.py \
  --device cuda:N
```

The script independently checks the selected GPU before loading tensors and
rejects a busy device. Fold-, dimension-, view-, and seed-level work is
checkpointed under `outputs/e2/e2_a01/checkpoints/`.
