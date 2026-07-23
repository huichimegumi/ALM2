# E1.6 rubric attribution controls

E1.6 tests whether the E1 criterion features use question-specific rubric
semantics, and whether they add information beyond a strong no-rubric baseline.
The frozen chunk embeddings and the unbounded structure-preserving chunk set are
reused unchanged.

## Experiments

The matched query uses the E1.2 format: retrieval instruction, task prompt,
criterion name, and criterion explanation. Four query ablations isolate those
components:

- `criterion_only` removes the task prompt;
- `prompt_only` removes criterion-specific text;
- `generic_dimension` replaces every criterion with one fixed description per dimension;
- `matched_full_no_instruction` removes the retrieval instruction.

Five deterministic cyclic derangements pair every report question with another
question's rubric. Each derangement is a permutation and has no self-matches.
The mismatch models are retrained under outer LOQO, and their five predictions
are also averaged into a conservative mismatch ensemble.

The matched representation evaluates the complete Ridge factorial:

```text
structure
global
rubric
global + structure
global + rubric
rubric + structure
global + rubric + structure
```

All variants preserve E1.4's nested LOQO protocol. Scaling, imputation, alpha
selection, and fitting occur only on outer-training questions. Official
pairwise accuracy of the weighted total is primary; dimension accuracy and
Spearman over comprehensiveness and instruction following are mechanism
diagnostics.

## Run

First encode only the rubric query variants. Check all GPUs and select an idle
one immediately before running this command:

```bash
nvidia-smi
.venv-system-python-backup/bin/python scripts/run_e1_query_variants.py \
  --device cuda:N \
  --max-length 4096 \
  --batch-size 8
```

This step encodes 210 queries per variant and does not re-encode document chunks.
The script independently verifies that the selected GPU is idle immediately
before allocating the model.

Then build the CPU-only controls and run nested LOQO Ridge:

```bash
.venv-system-python-backup/bin/python scripts/run_e1_6.py \
  --output-dir outputs/e1/e1_6_accuracy
```

Outputs are written under `outputs/e1/e1_6/`. They include all out-of-fold
predictions, model and per-question metrics, ordinary and alignment-specific
paired question bootstraps, selected Ridge hyperparameters, exact mismatch
mappings, cached control features, protocol metadata, and an automatic report.

## Interpretation

The primary incremental comparison is:

```text
matched_full_all > matched_full_global_structure
```

Criterion specificity additionally requires matched queries to outperform
generic-dimension and mismatched-rubric controls on comprehensiveness and
instruction following. A query control is not evidence of specificity merely
because it performs well in isolation; comparisons must use paired held-out
question results.
