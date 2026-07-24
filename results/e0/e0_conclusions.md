# AEOLLM-2 E0 conclusions

All learned results are out-of-fold. The primary protocol is leave-one-question-out;
`random_split_surface_ridge` is a deliberately leaky diagnostic and is not a valid final result.
Inside each outer training split, grouped validation selects hyperparameters by
pairwise accuracy, then Spearman, then MAE as deterministic tie-breakers.

Integrity gate: PASS (0 errors).

## Observed findings

- Surface-only LOQO reaches Accuracy 0.6675, Spearman 0.4547, and Kendall 0.3350; superficial document properties contain signal but do not explain ODAT.
- Generator/prompt metadata alone reaches Accuracy 0.7183. This is evidence of source/prompt confounding and must remain a diagnostic rather than a final evaluator.
- Adding metadata to surface features changes Accuracy from 0.6675 to 0.7150; see the paired question bootstrap below for uncertainty.
- Affine LOQO calibration reduces ODAT weighted-total MAE from 1.2919 to 0.4168 (67.7% reduction), while its three official ranking metrics are unchanged at displayed precision. ODAT therefore has a large scale bias that simple calibration can fix, but calibration does not improve its ordering.
- Random document splitting changes surface-model MAE from 0.5958 (LOQO) to 0.5942. Its rank metrics are not directly interpretable because documents within a question are scored by different fold models; LOQO remains the sole primary protocol.

## Main results

| model                       |   accuracy |   spearman |   kendall |    mae |   rmse |
|:----------------------------|-----------:|-----------:|----------:|-------:|-------:|
| odat_affine_loqo            |     0.8292 |     0.8082 |    0.6583 | 0.4168 | 0.5253 |
| odat_raw                    |     0.8292 |     0.8082 |    0.6583 | 1.2919 | 1.5702 |
| odat_multioutput_ridge_loqo |     0.8283 |     0.8088 |    0.6567 | 0.4260 | 0.5323 |
| odat_isotonic_loqo          |     0.8192 |     0.8065 |    0.6515 | 0.4111 | 0.5049 |
| metadata_ridge_loqo         |     0.7183 |     0.5953 |    0.4367 | 0.5301 | 0.6835 |
| surface_metadata_ridge_loqo |     0.7150 |     0.5756 |    0.4300 | 0.5986 | 0.7776 |
| surface_ridge_loqo          |     0.6675 |     0.4547 |    0.3350 | 0.5958 | 0.7579 |
| random_split_surface_ridge  |     0.6217 |     0.3521 |    0.2433 | 0.5942 | 0.7838 |
| mean_loqo                   |     0.0000 |   nan      |  nan      | 0.6478 | 0.8211 |

## Paired question bootstrap

| candidate                   | reference           | metric   |   mean_delta |   ci_low |   ci_high |   probability_delta_gt_zero |   positive_questions |   tied_questions |   net_correct_pairs |   n_questions |
|:----------------------------|:--------------------|:---------|-------------:|---------:|----------:|----------------------------:|---------------------:|-----------------:|--------------------:|--------------:|
| surface_ridge_loqo          | mean_loqo           | accuracy |       0.6675 |   0.6116 |    0.7242 |                      1.0000 |                   10 |                0 |            801.0000 |            10 |
| metadata_ridge_loqo         | mean_loqo           | accuracy |       0.7183 |   0.6700 |    0.7700 |                      1.0000 |                   10 |                0 |            862.0000 |            10 |
| surface_metadata_ridge_loqo | surface_ridge_loqo  | accuracy |       0.0475 |   0.0167 |    0.0817 |                      0.9990 |                    9 |                0 |             57.0000 |            10 |
| surface_metadata_ridge_loqo | surface_ridge_loqo  | spearman |       0.1209 |   0.0556 |    0.1953 |                      1.0000 |                    9 |                0 |            nan      |            10 |
| surface_metadata_ridge_loqo | surface_ridge_loqo  | kendall  |       0.0950 |   0.0333 |    0.1633 |                      0.9998 |                    9 |                0 |            nan      |            10 |
| odat_raw                    | surface_ridge_loqo  | accuracy |       0.1617 |   0.1208 |    0.1975 |                      1.0000 |                   10 |                0 |            194.0000 |            10 |
| odat_raw                    | surface_ridge_loqo  | spearman |       0.3535 |   0.2512 |    0.4471 |                      1.0000 |                   10 |                0 |            nan      |            10 |
| odat_raw                    | surface_ridge_loqo  | kendall  |       0.3233 |   0.2450 |    0.3967 |                      1.0000 |                   10 |                0 |            nan      |            10 |
| odat_raw                    | metadata_ridge_loqo | accuracy |       0.1108 |   0.0633 |    0.1625 |                      1.0000 |                    9 |                0 |            133.0000 |            10 |
| odat_raw                    | metadata_ridge_loqo | spearman |       0.2129 |   0.1150 |    0.3271 |                      1.0000 |                    9 |                0 |            nan      |            10 |
| odat_raw                    | metadata_ridge_loqo | kendall  |       0.2217 |   0.1250 |    0.3267 |                      1.0000 |                    9 |                0 |            nan      |            10 |
| odat_affine_loqo            | odat_raw            | accuracy |       0.0000 |  -0.0058 |    0.0058 |                      0.4404 |                    4 |                2 |              0.0000 |            10 |
| odat_affine_loqo            | odat_raw            | spearman |      -0.0000 |  -0.0159 |    0.0129 |                      0.5072 |                    5 |                1 |            nan      |            10 |
| odat_affine_loqo            | odat_raw            | kendall  |      -0.0000 |  -0.0117 |    0.0117 |                      0.4422 |                    4 |                2 |            nan      |            10 |
| odat_isotonic_loqo          | odat_raw            | accuracy |      -0.0100 |  -0.0242 |    0.0042 |                      0.0778 |                    2 |                2 |            -12.0000 |            10 |
| odat_isotonic_loqo          | odat_raw            | spearman |      -0.0017 |  -0.0201 |    0.0195 |                      0.4226 |                    4 |                0 |            nan      |            10 |
| odat_isotonic_loqo          | odat_raw            | kendall  |      -0.0069 |  -0.0342 |    0.0243 |                      0.3118 |                    2 |                2 |            nan      |            10 |
| odat_multioutput_ridge_loqo | odat_raw            | accuracy |      -0.0008 |  -0.0100 |    0.0092 |                      0.3820 |                    3 |                2 |             -1.0000 |            10 |
| odat_multioutput_ridge_loqo | odat_raw            | spearman |       0.0006 |  -0.0185 |    0.0159 |                      0.5350 |                    5 |                0 |            nan      |            10 |
| odat_multioutput_ridge_loqo | odat_raw            | kendall  |      -0.0017 |  -0.0200 |    0.0200 |                      0.4146 |                    3 |                2 |            nan      |            10 |

## Interpretation rules

- Prefer LOQO results over random document splits.
- Treat metadata-only performance as generator/prompt confounding, not evaluation ability.
- Calibration improving MAE without rank metrics indicates scale bias rather than ranking improvement.
- With only 10 independent questions, use the question-bootstrap intervals rather than pair-level p-values.
