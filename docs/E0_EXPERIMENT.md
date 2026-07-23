# AEOLLM-2 E0 experiment

E0 audits the data and official metric contract, measures surface/metadata
confounding, and tests whether post-hoc calibration explains the existing ODAT
result. It does not call a judge LLM and does not use a GPU.

## Inputs

The default command expects the workspace layout already present on Tyan:

- labels: `data/official/hf-aeollm/aeollm-2-train/train_deepresearch.csv`
- reports: `data/incoming/google-drive/train/Report*/Doc_*.docx`
- rubrics: `data/official/hf-aeollm/aeollm-2-train/rubric_dataset`
- generation mapping: `legacy/aeollm2_train_code/prompts/mapping_key_Readability.xlsx`
- legacy judge outputs: `legacy/aeollm2_train_code/outputs`

Every path has a CLI override. No experiment logic depends on the historical
Windows paths embedded in legacy result files.

## Environment

Use a CPU environment containing `requirements-e0.txt`. On the current Tyan
workspace, `.venv-system-python-backup` contains these packages:

```bash
.venv-system-python-backup/bin/python scripts/run_e0.py \
  --output-dir outputs/e0_accuracy
```

The root `.venv` is intended for model serving and may not contain pandas or
scikit-learn. E0 deliberately does not start vLLM, inspect GPU state, or set
`CUDA_VISIBLE_DEVICES`.

For a quick smoke run with shorter confidence intervals:

```bash
.venv-system-python-backup/bin/python scripts/run_e0.py \
  --bootstrap-resamples 100
```

## Protocol

- Primary split: Leave-One-Question-Out (10 folds).
- Inner hyperparameter selection: GroupKFold by question, maximizing pairwise
  accuracy, then Spearman, then minimizing MAE.
- Primary metric: exact official pairwise Accuracy of the weighted total.
- Secondary metrics: Spearman and Kendall.
- Spearman/Kendall aggregation: macro mean across questions.
- Accuracy aggregation: total correct pairs divided by total pairs.
- Accuracy uncertainty resamples questions and recomputes pooled correct/total
  pairs; document pairs are not treated as independent.
- Prediction gate: exactly the 160 labelled keys, unique keys, four finite
  dimension scores per key, and every score in `[0, 10]`.

The official evaluator is intentionally kept as the metric definition. The E0
validator is stricter because the official script silently inner-joins missing
documents, accepts partial dimensions, and evaluates partial question sets.

## Models and diagnostics

- `mean_loqo`: training-question mean for each dimension.
- `surface_ridge_loqo`: DOCX length, structure, table, citation, language, and
  repetition features.
- `metadata_ridge_loqo`: generation model ID and prompt variant only. This is a
  confounding diagnostic, not a deployable evaluator.
- `surface_metadata_ridge_loqo`: combined diagnostic.
- `random_split_surface_ridge`: deliberately invalid random document split,
  used only to quantify split leakage.
- `odat_raw`: frozen best legacy ODAT predictions.
- `odat_affine_loqo`, `odat_isotonic_loqo`, and
  `odat_multioutput_ridge_loqo`: calibration fitted only on the other nine
  questions.

## Outputs

The default output directory is `outputs/e0/`:

```text
protocol.yaml
data_manifest.csv
integrity_report.md
surface_features.csv
legacy_predictions_index.csv
legacy_recomputed_metrics.csv
official_metric_parity.json
selected_hyperparameters.csv
predictions/*.tsv
model_metrics.csv
split_diagnostic.csv
per_question_metrics.csv
bootstrap_ci.csv
paired_bootstrap.csv
e0_conclusions.md
run_status.json
```

`protocol.yaml` records input hashes, package versions, seed, and paths. The
manifest records report/rubric hashes so a later run can identify changed data.

## Tests

```bash
.venv-system-python-backup/bin/python -m pytest -q
```

Tests cover official tie semantics, strict completeness/range checks, perfect
metric agreement, XLSX mapping parsing, DOCX extraction, and complete LOQO
out-of-fold prediction generation.

## GPU rule for later experiments

E0 is CPU-only. For E1 and later GPU work, first inspect current GPU processes
and memory with `nvidia-smi`; explicitly select only an idle GPU with
`CUDA_VISIBLE_DEVICES`. If no GPU is idle, wait instead of sharing an occupied
card.
