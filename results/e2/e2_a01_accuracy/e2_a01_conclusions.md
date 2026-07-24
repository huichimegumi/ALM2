# E2-A0.1 dimension-separated diagonal diagnostic

E2-A0.1 changes one architectural assumption from E2-A0: comprehensiveness
and instruction following receive independent 1024-dimensional diagonal
metrics. The two heads remain jointly trained with the same loss. Encoder,
chunks, queries, pooling, optimizer, epochs, seeds, baseline, and splits are fixed.

## Main results

| model                                        |   accuracy |   spearman |   kendall |   accuracy_comprehensiveness |   accuracy_instruction_following |    mae |
|:---------------------------------------------|-----------:|-----------:|----------:|-----------------------------:|---------------------------------:|-------:|
| diagonal_mismatch_shift1_separate_hybrid     |     0.6925 |     0.5291 |    0.3850 |                       0.6550 |                           0.4958 | 0.7088 |
| diagonal_generic_separate_hybrid             |     0.6908 |     0.5238 |    0.3817 |                       0.6467 |                           0.4542 | 0.7068 |
| fixed_matched_hybrid                         |     0.6908 |     0.5226 |    0.3817 |                       0.6492 |                           0.4408 | 0.7061 |
| ridge_global_structure                       |     0.6908 |     0.5226 |    0.3817 |                       0.6500 |                           0.4392 | 0.7079 |
| diagonal_mismatch_shift5_separate_hybrid     |     0.6892 |     0.5206 |    0.3783 |                       0.6583 |                           0.4775 | 0.7010 |
| diagonal_mismatch_shift4_separate_hybrid     |     0.6892 |     0.5194 |    0.3783 |                       0.6483 |                           0.5417 | 0.7092 |
| diagonal_mismatched_ensemble_separate_hybrid |     0.6892 |     0.5194 |    0.3783 |                       0.6525 |                           0.5333 | 0.7050 |
| diagonal_mismatch_shift2_separate_hybrid     |     0.6883 |     0.5215 |    0.3767 |                       0.6592 |                           0.4675 | 0.7069 |
| diagonal_mismatch_shift3_separate_hybrid     |     0.6833 |     0.5135 |    0.3667 |                       0.6433 |                           0.5025 | 0.7036 |
| diagonal_shared_a0_hybrid                    |     0.6808 |     0.5006 |    0.3617 |                       0.6283 |                           0.4492 | 0.7256 |
| diagonal_matched_separate_hybrid             |     0.6808 |     0.4959 |    0.3617 |                       0.6342 |                           0.5208 | 0.7234 |

## Per-dimension learned-versus-fixed gates

| dimension             |   learned_minus_fixed_accuracy |   learned_fixed_ci_low |   learned_fixed_ci_high |   probability_delta_gt_zero |   positive_questions |   finite_questions |   learned_minus_baseline_accuracy | passed   |
|:----------------------|-------------------------------:|-----------------------:|------------------------:|----------------------------:|---------------------:|-------------------:|----------------------------------:|:---------|
| comprehensiveness     |                        -0.0150 |                -0.0300 |                 -0.0008 |                      0.0208 |                    2 |                 10 |                           -0.0158 | False    |
| instruction_following |                         0.0800 |                -0.0084 |                  0.2108 |                      0.9010 |                    3 |                 10 |                            0.0817 | True     |

A dimension passes only when learned minus fixed is positive, bootstrap
P(delta > 0) is at least 0.90, and the learned branch is not worse than
global + structure on mean dimension accuracy.

## Primary official-accuracy comparisons

| candidate                        | reference                                    | metric   |   mean_delta |   ci_low |   ci_high |   probability_delta_gt_zero |   positive_questions |   tied_questions |   net_correct_pairs |   n_questions |
|:---------------------------------|:---------------------------------------------|:---------|-------------:|---------:|----------:|----------------------------:|---------------------:|-----------------:|--------------------:|--------------:|
| diagonal_matched_separate_hybrid | fixed_matched_hybrid                         | accuracy |      -0.0100 |  -0.0283 |    0.0033 |                      0.0886 |                    2 |                4 |            -12.0000 |            10 |
| diagonal_matched_separate_hybrid | ridge_global_structure                       | accuracy |      -0.0100 |  -0.0292 |    0.0033 |                      0.0936 |                    2 |                4 |            -12.0000 |            10 |
| fixed_matched_hybrid             | ridge_global_structure                       | accuracy |       0.0000 |   0.0000 |    0.0000 |                      0.0000 |                    0 |               10 |              0.0000 |            10 |
| diagonal_matched_separate_hybrid | diagonal_shared_a0_hybrid                    | accuracy |       0.0000 |  -0.0058 |    0.0067 |                      0.4260 |                    1 |                7 |              0.0000 |            10 |
| diagonal_matched_separate_hybrid | diagonal_generic_separate_hybrid             | accuracy |      -0.0100 |  -0.0292 |    0.0067 |                      0.1174 |                    2 |                3 |            -12.0000 |            10 |
| diagonal_matched_separate_hybrid | diagonal_mismatched_ensemble_separate_hybrid | accuracy |      -0.0083 |  -0.0258 |    0.0033 |                      0.1120 |                    2 |                4 |            -10.0000 |            10 |

## Paired question bootstrap

| candidate                        | reference                                    | metric                         |   mean_delta |   ci_low |   ci_high |   probability_delta_gt_zero |   positive_questions |   tied_questions |   n_questions |
|:---------------------------------|:---------------------------------------------|:-------------------------------|-------------:|---------:|----------:|----------------------------:|---------------------:|-----------------:|--------------:|
| diagonal_matched_separate_hybrid | fixed_matched_hybrid                         | accuracy_comprehensiveness     |      -0.0150 |  -0.0300 |   -0.0008 |                      0.0208 |                    2 |                1 |            10 |
| diagonal_matched_separate_hybrid | fixed_matched_hybrid                         | accuracy_instruction_following |       0.0800 |  -0.0084 |    0.2108 |                      0.9010 |                    3 |                4 |            10 |
| diagonal_matched_separate_hybrid | fixed_matched_hybrid                         | spearman_comprehensiveness     |      -0.0314 |  -0.0652 |   -0.0009 |                      0.0218 |                    2 |                0 |            10 |
| diagonal_matched_separate_hybrid | fixed_matched_hybrid                         | spearman_instruction_following |      -0.0383 |  -0.0922 |    0.0032 |                      0.0410 |                    1 |                4 |             9 |
| diagonal_matched_separate_hybrid | ridge_global_structure                       | accuracy_comprehensiveness     |      -0.0158 |  -0.0308 |   -0.0017 |                      0.0134 |                    2 |                1 |            10 |
| diagonal_matched_separate_hybrid | ridge_global_structure                       | accuracy_instruction_following |       0.0817 |  -0.0058 |    0.2092 |                      0.9098 |                    3 |                4 |            10 |
| diagonal_matched_separate_hybrid | ridge_global_structure                       | spearman_comprehensiveness     |      -0.0349 |  -0.0646 |   -0.0055 |                      0.0086 |                    1 |                0 |            10 |
| diagonal_matched_separate_hybrid | ridge_global_structure                       | spearman_instruction_following |      -0.0335 |  -0.0858 |    0.0036 |                      0.0472 |                    1 |                4 |             9 |
| fixed_matched_hybrid             | ridge_global_structure                       | accuracy_comprehensiveness     |      -0.0008 |  -0.0025 |    0.0000 |                      0.0000 |                    0 |                9 |            10 |
| fixed_matched_hybrid             | ridge_global_structure                       | accuracy_instruction_following |       0.0017 |   0.0000 |    0.0050 |                      0.6466 |                    1 |                9 |            10 |
| fixed_matched_hybrid             | ridge_global_structure                       | spearman_comprehensiveness     |      -0.0035 |  -0.0106 |    0.0000 |                      0.0000 |                    0 |                9 |            10 |
| fixed_matched_hybrid             | ridge_global_structure                       | spearman_instruction_following |       0.0048 |   0.0000 |    0.0145 |                      0.6520 |                    1 |                8 |             9 |
| diagonal_matched_separate_hybrid | diagonal_shared_a0_hybrid                    | accuracy_comprehensiveness     |       0.0058 |  -0.0042 |    0.0167 |                      0.8384 |                    5 |                2 |            10 |
| diagonal_matched_separate_hybrid | diagonal_shared_a0_hybrid                    | accuracy_instruction_following |       0.0717 |  -0.0008 |    0.1725 |                      0.9720 |                    4 |                4 |            10 |
| diagonal_matched_separate_hybrid | diagonal_shared_a0_hybrid                    | spearman_comprehensiveness     |       0.0150 |  -0.0106 |    0.0412 |                      0.8580 |                    4 |                1 |            10 |
| diagonal_matched_separate_hybrid | diagonal_shared_a0_hybrid                    | spearman_instruction_following |       0.0665 |  -0.0567 |    0.2569 |                      0.7280 |                    3 |                4 |            10 |
| diagonal_matched_separate_hybrid | diagonal_generic_separate_hybrid             | accuracy_comprehensiveness     |      -0.0125 |  -0.0342 |    0.0100 |                      0.1362 |                    3 |                0 |            10 |
| diagonal_matched_separate_hybrid | diagonal_generic_separate_hybrid             | accuracy_instruction_following |       0.0667 |  -0.0358 |    0.2117 |                      0.8506 |                    3 |                5 |            10 |
| diagonal_matched_separate_hybrid | diagonal_generic_separate_hybrid             | spearman_comprehensiveness     |      -0.0270 |  -0.0830 |    0.0326 |                      0.1778 |                    3 |                0 |            10 |
| diagonal_matched_separate_hybrid | diagonal_generic_separate_hybrid             | spearman_instruction_following |      -0.0797 |  -0.2030 |    0.0072 |                      0.0876 |                    2 |                4 |             9 |
| diagonal_matched_separate_hybrid | diagonal_mismatched_ensemble_separate_hybrid | accuracy_comprehensiveness     |      -0.0183 |  -0.0400 |    0.0033 |                      0.0496 |                    3 |                1 |            10 |
| diagonal_matched_separate_hybrid | diagonal_mismatched_ensemble_separate_hybrid | accuracy_instruction_following |      -0.0125 |  -0.0575 |    0.0167 |                      0.3452 |                    3 |                5 |            10 |
| diagonal_matched_separate_hybrid | diagonal_mismatched_ensemble_separate_hybrid | spearman_comprehensiveness     |      -0.0465 |  -0.0961 |   -0.0021 |                      0.0214 |                    3 |                2 |            10 |
| diagonal_matched_separate_hybrid | diagonal_mismatched_ensemble_separate_hybrid | spearman_instruction_following |      -0.0404 |  -0.1651 |    0.0369 |                      0.3298 |                    3 |                4 |            10 |

## Decision

- `comprehensiveness`: FAIL; learned-fixed accuracy=-0.0150, P(delta>0)=0.0208, 2/10 finite questions positive, learned-baseline accuracy=-0.0158.
- `instruction_following`: PASS; learned-fixed accuracy=+0.0800, P(delta>0)=0.9010, 3/10 finite questions positive, learned-baseline accuracy=+0.0817.
- Separated minus shared `comprehensiveness` accuracy: +0.0058.
- Separated minus shared `instruction_following` accuracy: +0.0717.
- Mean final training Huber: 0.0113 learned versus 0.0436 fixed; the lower training loss and worse LOQO ranking indicate overfitting.
- Original shared E2-A0 predictions loaded: yes.
- Conditional generic/mismatch controls were run for: `instruction_following`.
- This experiment diagnoses negative transfer only. It does not change
  supervision or establish that the learned score measures satisfaction.
- The continuation condition for criterion-only E2-A0.1b was not met;
  it should not be run as the next confirmatory document-level experiment.
