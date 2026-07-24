# Accuracy-first experiment summary

| experiment   | accuracy_leader                          |   accuracy |   correct_pairs |   spearman |   kendall |
|:-------------|:-----------------------------------------|-----------:|----------------:|-----------:|----------:|
| E0           | odat_affine_loqo                         |     0.8292 |             995 |     0.8082 |    0.6583 |
| E1.4         | rubric_structure_unbounded               |     0.7050 |             846 |     0.5424 |    0.4100 |
| E1.5         | ridge_all_unbounded                      |     0.7008 |             841 |     0.5453 |    0.4017 |
| E1.6         | criterion_only_rubric                    |     0.7142 |             857 |     0.5662 |    0.4283 |
| E1.7         | matched_full_selective                   |     0.7067 |             848 |     0.5615 |    0.4133 |
| E2-A0        | diagonal_mismatch_shift1_hybrid          |     0.6925 |             831 |     0.5359 |    0.3850 |
| E2-A0.1      | diagonal_mismatch_shift1_separate_hybrid |     0.6925 |             831 |     0.5291 |    0.3850 |

See `docs/ACCURACY_FIRST_PROTOCOL.md` for the evaluation protocol.
