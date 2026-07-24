# E1.4 nested LOQO Ridge results

All predictions are outer leave-one-question-out. Prospective runs select
Ridge alpha inside each outer training split using grouped pairwise accuracy,
with Spearman and then MAE as deterministic tie-breaks. A retrospective note,
when present, means the displayed saved predictions retain historical selection.
E1.5 ranking losses are not used here.

## Main results

| model                      |   accuracy |   spearman |   kendall |    mae |   rmse |
|:---------------------------|-----------:|-----------:|----------:|-------:|-------:|
| rubric_structure_unbounded |     0.7050 |     0.5424 |    0.4100 | 0.6402 | 0.8028 |
| all_unbounded              |     0.7008 |     0.5453 |    0.4017 | 0.6167 | 0.7647 |
| all_capped                 |     0.7000 |     0.5403 |    0.4000 | 0.6285 | 0.7686 |
| rubric_structure_capped    |     0.6867 |     0.5029 |    0.3733 | 0.6268 | 0.7821 |
| structure                  |     0.6675 |     0.4547 |    0.3350 | 0.5958 | 0.7579 |
| rubric_capped              |     0.6642 |     0.4488 |    0.3283 | 0.6303 | 0.7972 |
| global_unbounded           |     0.6625 |     0.4594 |    0.3250 | 0.7250 | 0.9061 |
| global_capped              |     0.6592 |     0.4394 |    0.3183 | 0.7223 | 0.8755 |
| rubric_unbounded           |     0.6542 |     0.4097 |    0.3083 | 0.6403 | 0.7909 |
| mean_loqo                  |     0.0000 |   nan      |  nan      | 0.6478 | 0.8211 |

## Primary checks

- capped: rubric minus global accuracy = +0.0050.
- capped: all minus rubric accuracy = +0.0358.
- unbounded: rubric minus global accuracy = -0.0083.
- unbounded: all minus rubric accuracy = +0.0467.

## Paired question bootstrap

| candidate                  | reference        | metric   |   mean_delta |   ci_low |   ci_high |   probability_delta_gt_zero |   positive_questions |   tied_questions |   net_correct_pairs |   n_questions |
|:---------------------------|:-----------------|:---------|-------------:|---------:|----------:|----------------------------:|---------------------:|-----------------:|--------------------:|--------------:|
| rubric_capped              | global_capped    | accuracy |       0.0050 |  -0.0350 |    0.0467 |                      0.5822 |                    5 |                1 |              6.0000 |            10 |
| rubric_capped              | global_capped    | spearman |       0.0094 |  -0.0871 |    0.1121 |                      0.5630 |                    5 |                0 |            nan      |            10 |
| rubric_capped              | global_capped    | kendall  |       0.0100 |  -0.0667 |    0.0933 |                      0.5974 |                    5 |                1 |            nan      |            10 |
| rubric_capped              | structure        | accuracy |      -0.0033 |  -0.0483 |    0.0475 |                      0.4292 |                    3 |                2 |             -4.0000 |            10 |
| rubric_capped              | structure        | spearman |      -0.0059 |  -0.1309 |    0.1330 |                      0.4608 |                    5 |                1 |            nan      |            10 |
| rubric_capped              | structure        | kendall  |      -0.0067 |  -0.0983 |    0.0933 |                      0.4436 |                    3 |                2 |            nan      |            10 |
| rubric_structure_capped    | structure        | accuracy |       0.0192 |  -0.0150 |    0.0550 |                      0.8456 |                    4 |                1 |             23.0000 |            10 |
| rubric_structure_capped    | structure        | spearman |       0.0482 |  -0.0541 |    0.1559 |                      0.8132 |                    6 |                0 |            nan      |            10 |
| rubric_structure_capped    | structure        | kendall  |       0.0383 |  -0.0300 |    0.1117 |                      0.8386 |                    4 |                1 |            nan      |            10 |
| all_capped                 | rubric_capped    | accuracy |       0.0358 |   0.0042 |    0.0692 |                      0.9864 |                    7 |                0 |             43.0000 |            10 |
| all_capped                 | rubric_capped    | spearman |       0.0915 |   0.0126 |    0.1779 |                      0.9914 |                    7 |                0 |            nan      |            10 |
| all_capped                 | rubric_capped    | kendall  |       0.0717 |   0.0050 |    0.1400 |                      0.9836 |                    7 |                0 |            nan      |            10 |
| all_capped                 | structure        | accuracy |       0.0325 |  -0.0108 |    0.0758 |                      0.9254 |                    6 |                0 |             39.0000 |            10 |
| all_capped                 | structure        | spearman |       0.0856 |  -0.0132 |    0.1927 |                      0.9536 |                    6 |                0 |            nan      |            10 |
| all_capped                 | structure        | kendall  |       0.0650 |  -0.0217 |    0.1500 |                      0.9292 |                    6 |                0 |            nan      |            10 |
| rubric_unbounded           | global_unbounded | accuracy |      -0.0083 |  -0.0367 |    0.0292 |                      0.2756 |                    2 |                1 |            -10.0000 |            10 |
| rubric_unbounded           | global_unbounded | spearman |      -0.0497 |  -0.1441 |    0.0638 |                      0.1628 |                    1 |                0 |            nan      |            10 |
| rubric_unbounded           | global_unbounded | kendall  |      -0.0167 |  -0.0733 |    0.0567 |                      0.2884 |                    2 |                1 |            nan      |            10 |
| rubric_unbounded           | structure        | accuracy |      -0.0133 |  -0.0475 |    0.0233 |                      0.2356 |                    3 |                2 |            -16.0000 |            10 |
| rubric_unbounded           | structure        | spearman |      -0.0450 |  -0.1500 |    0.0618 |                      0.2158 |                    5 |                0 |            nan      |            10 |
| rubric_unbounded           | structure        | kendall  |      -0.0267 |  -0.0950 |    0.0433 |                      0.2410 |                    3 |                2 |            nan      |            10 |
| rubric_structure_unbounded | structure        | accuracy |       0.0375 |  -0.0008 |    0.0733 |                      0.9694 |                    6 |                0 |             45.0000 |            10 |
| rubric_structure_unbounded | structure        | spearman |       0.0876 |  -0.0159 |    0.1850 |                      0.9552 |                    6 |                0 |            nan      |            10 |
| rubric_structure_unbounded | structure        | kendall  |       0.0750 |  -0.0000 |    0.1483 |                      0.9736 |                    6 |                0 |            nan      |            10 |
| all_unbounded              | rubric_unbounded | accuracy |       0.0467 |   0.0208 |    0.0725 |                      0.9996 |                    9 |                0 |             56.0000 |            10 |
| all_unbounded              | rubric_unbounded | spearman |       0.1356 |   0.0647 |    0.2068 |                      1.0000 |                    9 |                0 |            nan      |            10 |
| all_unbounded              | rubric_unbounded | kendall  |       0.0933 |   0.0417 |    0.1450 |                      0.9998 |                    9 |                0 |            nan      |            10 |
| all_unbounded              | structure        | accuracy |       0.0333 |   0.0000 |    0.0675 |                      0.9724 |                    7 |                0 |             40.0000 |            10 |
| all_unbounded              | structure        | spearman |       0.0906 |   0.0079 |    0.1800 |                      0.9840 |                    6 |                0 |            nan      |            10 |
| all_unbounded              | structure        | kendall  |       0.0667 |  -0.0017 |    0.1333 |                      0.9720 |                    7 |                0 |            nan      |            10 |
| global_unbounded           | global_capped    | accuracy |       0.0033 |  -0.0233 |    0.0267 |                      0.5988 |                    6 |                0 |              4.0000 |            10 |
| global_unbounded           | global_capped    | spearman |       0.0200 |  -0.0341 |    0.0715 |                      0.7734 |                    6 |                0 |            nan      |            10 |
| global_unbounded           | global_capped    | kendall  |       0.0067 |  -0.0483 |    0.0550 |                      0.6094 |                    6 |                0 |            nan      |            10 |
| rubric_unbounded           | rubric_capped    | accuracy |      -0.0100 |  -0.0300 |    0.0067 |                      0.1302 |                    4 |                1 |            -12.0000 |            10 |
| rubric_unbounded           | rubric_capped    | spearman |      -0.0391 |  -0.0962 |    0.0085 |                      0.0620 |                    3 |                0 |            nan      |            10 |
| rubric_unbounded           | rubric_capped    | kendall  |      -0.0200 |  -0.0583 |    0.0133 |                      0.1420 |                    4 |                1 |            nan      |            10 |
| all_unbounded              | all_capped       | accuracy |       0.0008 |  -0.0192 |    0.0167 |                      0.5544 |                    7 |                0 |              1.0000 |            10 |
| all_unbounded              | all_capped       | spearman |       0.0050 |  -0.0315 |    0.0391 |                      0.6206 |                    7 |                0 |            nan      |            10 |
| all_unbounded              | all_capped       | kendall  |       0.0017 |  -0.0383 |    0.0333 |                      0.5720 |                    7 |                0 |            nan      |            10 |

## Interpretation

- `rubric_*` uses only the target dimension's primary criterion-conditioned features.
- `structure` reuses E0 surface features and excludes generator/prompt metadata.
- `all_*` combines global embeddings, target-dimension rubric features, and structure.
- Fixed top-3/top-5 and chunk-count diagnostic features are excluded.
- Accuracy is the primary metric; Spearman diagnoses large rank displacement.
- Kendall is reported as an accuracy-consistency check because the current
  weighted totals contain almost no ties.
- Prefer question-level paired uncertainty over treating pairs as independent.
