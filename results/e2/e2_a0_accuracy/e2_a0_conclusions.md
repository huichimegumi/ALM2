# E2-A0 minimal learned diagonal interaction

All models use frozen Qwen embeddings and unbounded chunks. The interaction
branch changes only comprehensiveness and instruction following; insight and
readability remain the outer-fold global+structure Ridge predictions.

## Main results

| model                               |   accuracy |   spearman |   kendall |   accuracy_comprehensiveness |   accuracy_instruction_following |    mae |
|:------------------------------------|-----------:|-----------:|----------:|-----------------------------:|---------------------------------:|-------:|
| diagonal_mismatch_shift1_hybrid     |     0.6925 |     0.5359 |    0.3850 |                       0.6517 |                           0.4975 | 0.7051 |
| diagonal_mismatched_ensemble_hybrid |     0.6908 |     0.5265 |    0.3817 |                       0.6500 |                           0.5083 | 0.7032 |
| fixed_matched_hybrid                |     0.6908 |     0.5226 |    0.3817 |                       0.6492 |                           0.4408 | 0.7061 |
| ridge_global_structure              |     0.6908 |     0.5226 |    0.3817 |                       0.6500 |                           0.4392 | 0.7079 |
| diagonal_generic_hybrid             |     0.6892 |     0.5241 |    0.3783 |                       0.6475 |                           0.4633 | 0.7037 |
| diagonal_mismatch_shift2_hybrid     |     0.6867 |     0.5171 |    0.3733 |                       0.6592 |                           0.4408 | 0.7099 |
| diagonal_mismatch_shift5_hybrid     |     0.6858 |     0.5188 |    0.3717 |                       0.6625 |                           0.4583 | 0.7078 |
| diagonal_mismatch_shift4_hybrid     |     0.6858 |     0.5135 |    0.3717 |                       0.6542 |                           0.4525 | 0.7069 |
| diagonal_mismatch_shift3_hybrid     |     0.6842 |     0.5150 |    0.3683 |                       0.6375 |                           0.4433 | 0.6927 |
| diagonal_matched_hybrid             |     0.6808 |     0.5006 |    0.3617 |                       0.6283 |                           0.4492 | 0.7256 |

## Primary official-accuracy comparisons

| candidate                       | reference                           | metric   |   mean_delta |   ci_low |   ci_high |   probability_delta_gt_zero |   positive_questions |   tied_questions |   net_correct_pairs |   n_questions |
|:--------------------------------|:------------------------------------|:---------|-------------:|---------:|----------:|----------------------------:|---------------------:|-----------------:|--------------------:|--------------:|
| diagonal_matched_hybrid         | fixed_matched_hybrid                | accuracy |      -0.0100 |  -0.0267 |    0.0025 |                      0.0674 |                    2 |                4 |            -12.0000 |            10 |
| diagonal_matched_hybrid         | ridge_global_structure              | accuracy |      -0.0100 |  -0.0267 |    0.0025 |                      0.0700 |                    2 |                4 |            -12.0000 |            10 |
| fixed_matched_hybrid            | ridge_global_structure              | accuracy |       0.0000 |   0.0000 |    0.0000 |                      0.0000 |                    0 |               10 |              0.0000 |            10 |
| diagonal_matched_hybrid         | diagonal_generic_hybrid             | accuracy |      -0.0083 |  -0.0217 |    0.0033 |                      0.0848 |                    2 |                4 |            -10.0000 |            10 |
| diagonal_matched_hybrid         | diagonal_mismatched_ensemble_hybrid | accuracy |      -0.0100 |  -0.0233 |    0.0008 |                      0.0318 |                    2 |                4 |            -12.0000 |            10 |
| diagonal_mismatch_shift2_hybrid | ridge_global_structure              | accuracy |      -0.0042 |  -0.0092 |    0.0008 |                      0.0290 |                    1 |                5 |             -5.0000 |            10 |

## Decision

- Learned diagonal versus fixed cosine: -0.0100 official accuracy, P(delta>0)=0.0674, -12 net correct pairs.
- Learned diagonal versus global+structure: -0.0100.
- Matched versus generic learned interaction: -0.0083.
- Matched versus mismatched learned ensemble: -0.0100.
- Mismatch shift 2 has the highest point accuracy but only -0.0042 over baseline (P(delta>0)=0.0290);
  because it uses the wrong rubric, it is a negative control rather than a candidate.
- E2-A0 learned-interaction gate: FAIL.

The gate uses the official weighted-total pairwise accuracy. Dimension accuracy
and Spearman remain mechanism diagnostics, and Kendall checks consistency.
