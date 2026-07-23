# E1.7 selective query Ridge

E1.7 tests a dimension-aware use of the frozen E1 representation. It retains
the full E1.3 criterion–chunk cosine summary instead of the four-statistic
pooling used by E2-A0.

The no-rubric baseline uses global embedding and structure features for all
four dimensions. Selective models add rubric features only to
comprehensiveness and instruction following; insight and readability remain
on the no-rubric branch.

Five models are evaluated:

- `global_structure`;
- `matched_full_selective`;
- `criterion_only_selective`;
- `fixed_dimension_selective`, with criterion-only for comprehensiveness and
  full prompt + criterion for instruction following;
- `nested_query_selective`, which chooses between the two query forms
  independently for both rubric-aligned dimensions inside every outer fold.

For the nested policy, query source and Ridge alpha are selected jointly by
minimum MAE under question-grouped inner cross-validation. The held-out
question is not used for selection, scaling, imputation, or fitting. The
primary research metric is the mean question-level Spearman over
comprehensiveness and instruction following. Official weighted-total results
remain secondary and are fully reported.

The fixed dimension mapping is an exploratory, researcher-informed analysis
motivated by E1.6. It must not be described as validation on untouched tasks.
The nested policy is the leakage-safe test of whether that kind of query choice
can be recovered from outer-training questions alone.

## Run

```bash
.venv-system-python-backup/bin/python scripts/run_e1_7.py
```

E1.7 uses scikit-learn Ridge on only 160 documents. Its numerical backend is
CPU-only, and moving this workload to a GPU would add transfer and framework
overhead rather than accelerate it. No embedding generation is needed because
the matched and criterion-only feature caches are reused unchanged.

Outputs are written to `outputs/e1/e1_7/`, including out-of-fold predictions,
all inner candidate scores, fold-level query and alpha choices, paired
question bootstraps, protocol metadata, and the automatic decision report.

The pre-specified gate compares `nested_query_selective` with
`global_structure`: positive mean alignment delta, bootstrap probability of a
positive delta at least 0.90, and at least 7 of 10 held-out questions positive.
