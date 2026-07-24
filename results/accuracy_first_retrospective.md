# Accuracy-first retrospective summary

> **Accuracy-first retrospective.** Saved predictions are unchanged and were produced under the historical protocol. Where historical inner selection used MAE or analysis emphasized Spearman, this report reinterprets the fixed out-of-fold predictions; it is not fresh confirmatory evidence.

| experiment   | accuracy_leader                 |   accuracy |   correct_pairs |   spearman |   kendall |
|:-------------|:--------------------------------|-----------:|----------------:|-----------:|----------:|
| E0           | odat_affine_loqo                |     0.8292 |             995 |     0.8082 |    0.6583 |
| E1.4         | all_unbounded                   |     0.6967 |             836 |     0.5391 |    0.3933 |
| E1.5         | ridge_all_unbounded             |     0.6967 |             836 |     0.5391 |    0.3933 |
| E1.6         | criterion_only_all              |     0.7083 |             850 |     0.5709 |    0.4167 |
| E1.7         | nested_query_selective          |     0.7092 |             851 |     0.5712 |    0.4183 |
| E2-A0        | diagonal_mismatch_shift2_hybrid |     0.7008 |             841 |     0.5418 |    0.4017 |
| E2-A0.1      | ridge_global_structure          |     0.6942 |             833 |     0.5400 |    0.3883 |

See `docs/ACCURACY_FIRST_PROTOCOL.md` for the prospective protocol.
