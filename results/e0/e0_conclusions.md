# AEOLLM-2 E0 conclusions

> **Accuracy-first retrospective.** Saved predictions are unchanged and were produced under the historical protocol. Where historical inner selection used MAE or analysis emphasized Spearman, this report reinterprets the fixed out-of-fold predictions; it is not fresh confirmatory evidence.

All learned results are out-of-fold. The primary protocol is leave-one-question-out;
`random_split_surface_ridge` is a deliberately leaky diagnostic and is not a valid final result.

Integrity gate: PASS (0 errors).

## Observed findings

- Surface-only LOQO reaches Accuracy 0.6725, Spearman 0.4632, and Kendall 0.3450; superficial document properties contain signal but do not explain ODAT.
- Generator/prompt metadata alone reaches Accuracy 0.7183. This is evidence of source/prompt confounding and must remain a diagnostic rather than a final evaluator.
- Adding metadata to surface features changes Accuracy from 0.6725 to 0.6908; see the paired question bootstrap below for uncertainty.
- Affine LOQO calibration reduces ODAT weighted-total MAE from 1.2919 to 0.4168 (67.7% reduction), while its three official ranking metrics are unchanged at displayed precision. ODAT therefore has a large scale bias that simple calibration can fix, but calibration does not improve its ordering.
- Random document splitting changes surface-model MAE from 0.6095 (LOQO) to 0.5942. Its rank metrics are not directly interpretable because documents within a question are scored by different fold models; LOQO remains the sole primary protocol.

## Main results

| model                       |   accuracy |   spearman |   kendall |    mae |   rmse |
|:----------------------------|-----------:|-----------:|----------:|-------:|-------:|
| odat_affine_loqo            |     0.8292 |     0.8082 |    0.6583 | 0.4168 | 0.5253 |
| odat_raw                    |     0.8292 |     0.8082 |    0.6583 | 1.2919 | 1.5702 |
| odat_multioutput_ridge_loqo |     0.8283 |     0.8088 |    0.6567 | 0.4233 | 0.5283 |
| odat_isotonic_loqo          |     0.8192 |     0.8065 |    0.6515 | 0.4111 | 0.5049 |
| metadata_ridge_loqo         |     0.7183 |     0.5953 |    0.4367 | 0.5296 | 0.6830 |
| surface_metadata_ridge_loqo |     0.6908 |     0.5047 |    0.3817 | 0.6017 | 0.7671 |
| surface_ridge_loqo          |     0.6725 |     0.4632 |    0.3450 | 0.6095 | 0.7770 |
| random_split_surface_ridge  |     0.6217 |     0.3521 |    0.2433 | 0.5942 | 0.7838 |
| mean_loqo                   |     0.0000 |   nan      |  nan      | 0.6478 | 0.8211 |

## Paired question bootstrap

| candidate                   | reference           | metric   |   mean_delta |   ci_low |   ci_high |   probability_delta_gt_zero |   positive_questions |   tied_questions |   net_correct_pairs |   n_questions |
|:----------------------------|:--------------------|:---------|-------------:|---------:|----------:|----------------------------:|---------------------:|-----------------:|--------------------:|--------------:|
| surface_ridge_loqo          | mean_loqo           | accuracy |       0.6725 |   0.6000 |    0.7392 |                      1.0000 |                   10 |                0 |            807.0000 |            10 |
| metadata_ridge_loqo         | mean_loqo           | accuracy |       0.7183 |   0.6700 |    0.7700 |                      1.0000 |                   10 |                0 |            862.0000 |            10 |
| surface_metadata_ridge_loqo | surface_ridge_loqo  | accuracy |       0.0183 |   0.0017 |    0.0367 |                      0.9836 |                    5 |                3 |             22.0000 |            10 |
| surface_metadata_ridge_loqo | surface_ridge_loqo  | spearman |       0.0415 |   0.0056 |    0.0818 |                      0.9908 |                    7 |                0 |            nan      |            10 |
| surface_metadata_ridge_loqo | surface_ridge_loqo  | kendall  |       0.0367 |   0.0033 |    0.0717 |                      0.9858 |                    5 |                3 |            nan      |            10 |
| odat_affine_loqo            | odat_raw            | accuracy |       0.0000 |  -0.0058 |    0.0058 |                      0.4394 |                    4 |                2 |              0.0000 |            10 |
| odat_affine_loqo            | odat_raw            | spearman |      -0.0000 |  -0.0159 |    0.0132 |                      0.5180 |                    5 |                1 |            nan      |            10 |
| odat_affine_loqo            | odat_raw            | kendall  |      -0.0000 |  -0.0117 |    0.0117 |                      0.4478 |                    4 |                2 |            nan      |            10 |
| odat_isotonic_loqo          | odat_raw            | accuracy |      -0.0100 |  -0.0250 |    0.0042 |                      0.0770 |                    2 |                2 |            -12.0000 |            10 |
| odat_isotonic_loqo          | odat_raw            | spearman |      -0.0017 |  -0.0200 |    0.0197 |                      0.4244 |                    4 |                0 |            nan      |            10 |
| odat_isotonic_loqo          | odat_raw            | kendall  |      -0.0069 |  -0.0342 |    0.0241 |                      0.3166 |                    2 |                2 |            nan      |            10 |
| odat_multioutput_ridge_loqo | odat_raw            | accuracy |      -0.0008 |  -0.0100 |    0.0092 |                      0.3818 |                    3 |                2 |             -1.0000 |            10 |
| odat_multioutput_ridge_loqo | odat_raw            | spearman |       0.0006 |  -0.0179 |    0.0156 |                      0.5370 |                    5 |                0 |            nan      |            10 |
| odat_multioutput_ridge_loqo | odat_raw            | kendall  |      -0.0017 |  -0.0200 |    0.0184 |                      0.4260 |                    3 |                2 |            nan      |            10 |
| odat_raw                    | surface_ridge_loqo  | accuracy |       0.1567 |   0.1150 |    0.1950 |                      1.0000 |                   10 |                0 |            188.0000 |            10 |
| odat_raw                    | surface_ridge_loqo  | spearman |       0.3450 |   0.2279 |    0.4591 |                      1.0000 |                   10 |                0 |            nan      |            10 |
| odat_raw                    | surface_ridge_loqo  | kendall  |       0.3133 |   0.2250 |    0.3867 |                      1.0000 |                   10 |                0 |            nan      |            10 |
| odat_raw                    | metadata_ridge_loqo | accuracy |       0.1108 |   0.0642 |    0.1625 |                      1.0000 |                    9 |                0 |            133.0000 |            10 |
| odat_raw                    | metadata_ridge_loqo | spearman |       0.2129 |   0.1165 |    0.3227 |                      1.0000 |                    9 |                0 |            nan      |            10 |
| odat_raw                    | metadata_ridge_loqo | kendall  |       0.2217 |   0.1267 |    0.3250 |                      1.0000 |                    9 |                0 |            nan      |            10 |

## Interpretation rules

- Prefer LOQO results over random document splits.
- Treat metadata-only performance as generator/prompt confounding, not evaluation ability.
- Calibration improving MAE without rank metrics indicates scale bias rather than ranking improvement.
- With only 10 independent questions, use the question-bootstrap intervals rather than pair-level p-values.
