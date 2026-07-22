# E1.5 fixed MLP with Huber and within-question pairwise loss

E1.5 tests whether an explicitly ranking-aware objective improves the frozen E1
representations. It compares identical four-head MLPs trained with Huber alone
against Huber plus weighted pairwise logistic loss.

## Fixed protocol

- Outer split: leave one complete question out.
- No validation on or early stopping against the held-out question.
- Architecture: `input -> 64 -> GELU -> dropout -> 16 -> GELU -> dropout -> 4`.
- Training: 150 full-batch AdamW epochs with fixed hyperparameters.
- Ensemble: three fixed random seeds averaged at prediction time.
- Targets are standardized using outer-training documents only.
- Input scaling and optional 64-component PCA are fitted on outer training only.
- PCA is used only when raw feature dimension is at least 128.
- Pair construction is restricted to documents from the same question and dimension.
- Score gaps at or below 0.1 are ignored; larger gaps weight the loss up to 2.0.
- Pair weights are mean-normalized and `lambda_pair = 0.5`.

The selected unbounded representation is used because E1.4 showed it was
materially better than capped for rubric-only features. Three feature groups are
tested so a pairwise gain can be separated from a generic structure-ranking gain:

- E0 structure features;
- all four dimensions' primary rubric features;
- global embedding + rubric + structure.

## Run

```bash
.venv-system-python-backup/bin/python scripts/run_e1_pairwise.py
```

The implementation is CPU-only and does not require a GPU availability check.
Outputs under `outputs/e1/e1_5/` include out-of-fold predictions, metrics,
training diagnostics for every outer fold and seed, question-level bootstrap
comparisons, the complete protocol, and a conclusions report.
