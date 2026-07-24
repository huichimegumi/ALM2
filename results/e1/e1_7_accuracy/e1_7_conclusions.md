# E1.7 selective query Ridge

E1.7 keeps the rich E1.3 criterion–chunk cosine summaries and asks whether
rubric evidence should be routed only to comprehensiveness and instruction
following. Insight and readability always use global + structure features.

## Main results

| model                     |   accuracy |   spearman |   kendall |   accuracy_comprehensiveness |   accuracy_instruction_following |    mae |
|:--------------------------|-----------:|-----------:|----------:|-----------------------------:|---------------------------------:|-------:|
| matched_full_selective    |     0.7067 |     0.5615 |    0.4133 |                       0.6525 |                           0.4967 | 0.6153 |
| nested_query_selective    |     0.7042 |     0.5591 |    0.4083 |                       0.6633 |                           0.4383 | 0.6285 |
| criterion_only_selective  |     0.7025 |     0.5600 |    0.4050 |                       0.6633 |                           0.5100 | 0.6800 |
| fixed_dimension_selective |     0.7017 |     0.5556 |    0.4033 |                       0.6633 |                           0.4967 | 0.5998 |
| global_structure          |     0.6908 |     0.5226 |    0.3817 |                       0.6500 |                           0.4392 | 0.7079 |

## Primary official-accuracy comparisons

| candidate                 | reference                 | metric   |   mean_delta |   ci_low |   ci_high |   probability_delta_gt_zero |   positive_questions |   tied_questions |   net_correct_pairs |   n_questions |
|:--------------------------|:--------------------------|:---------|-------------:|---------:|----------:|----------------------------:|---------------------:|-----------------:|--------------------:|--------------:|
| matched_full_selective    | global_structure          | accuracy |       0.0158 |  -0.0050 |    0.0392 |                      0.9138 |                    6 |                0 |             19.0000 |            10 |
| criterion_only_selective  | global_structure          | accuracy |       0.0117 |  -0.0025 |    0.0317 |                      0.9092 |                    5 |                2 |             14.0000 |            10 |
| fixed_dimension_selective | global_structure          | accuracy |       0.0108 |  -0.0058 |    0.0308 |                      0.8664 |                    5 |                2 |             13.0000 |            10 |
| nested_query_selective    | global_structure          | accuracy |       0.0133 |  -0.0058 |    0.0350 |                      0.8914 |                    5 |                2 |             16.0000 |            10 |
| fixed_dimension_selective | matched_full_selective    | accuracy |      -0.0050 |  -0.0175 |    0.0075 |                      0.2032 |                    3 |                3 |             -6.0000 |            10 |
| nested_query_selective    | fixed_dimension_selective | accuracy |       0.0025 |  -0.0017 |    0.0075 |                      0.7934 |                    2 |                7 |              3.0000 |            10 |
| nested_query_selective    | matched_full_selective    | accuracy |      -0.0025 |  -0.0150 |    0.0100 |                      0.3212 |                    3 |                2 |             -3.0000 |            10 |

## Dimension-mechanism diagnostics

| candidate                 | reference                 | metric             |   mean_delta |   ci_low |   ci_high |   probability_delta_gt_zero |   positive_questions |   tied_questions |   n_questions |
|:--------------------------|:--------------------------|:-------------------|-------------:|---------:|----------:|----------------------------:|---------------------:|-----------------:|--------------:|
| matched_full_selective    | global_structure          | alignment_accuracy |       0.0300 |  -0.0192 |    0.0871 |                      0.8700 |                    6 |                0 |            10 |
| matched_full_selective    | global_structure          | alignment_spearman |       0.0726 |  -0.0241 |    0.1760 |                      0.9294 |                    7 |                0 |            10 |
| criterion_only_selective  | global_structure          | alignment_accuracy |       0.0421 |  -0.0108 |    0.1133 |                      0.9246 |                    6 |                0 |            10 |
| criterion_only_selective  | global_structure          | alignment_spearman |       0.0742 |   0.0002 |    0.1534 |                      0.9754 |                    7 |                0 |            10 |
| fixed_dimension_selective | global_structure          | alignment_accuracy |       0.0354 |  -0.0200 |    0.1017 |                      0.8540 |                    5 |                0 |            10 |
| fixed_dimension_selective | global_structure          | alignment_spearman |       0.0744 |  -0.0238 |    0.1918 |                      0.9160 |                    6 |                0 |            10 |
| nested_query_selective    | global_structure          | alignment_accuracy |       0.0063 |  -0.0325 |    0.0496 |                      0.5924 |                    4 |                0 |            10 |
| nested_query_selective    | global_structure          | alignment_spearman |       0.0354 |  -0.0557 |    0.1515 |                      0.7152 |                    5 |                0 |            10 |
| fixed_dimension_selective | matched_full_selective    | alignment_accuracy |       0.0054 |  -0.0088 |    0.0192 |                      0.7812 |                    6 |                0 |            10 |
| fixed_dimension_selective | matched_full_selective    | alignment_spearman |       0.0018 |  -0.0437 |    0.0392 |                      0.5490 |                    6 |                0 |            10 |
| nested_query_selective    | fixed_dimension_selective | alignment_accuracy |      -0.0292 |  -0.0771 |    0.0008 |                      0.0812 |                    1 |                7 |            10 |
| nested_query_selective    | fixed_dimension_selective | alignment_spearman |      -0.0390 |  -0.0958 |    0.0020 |                      0.0876 |                    1 |                7 |            10 |
| nested_query_selective    | matched_full_selective    | alignment_accuracy |      -0.0237 |  -0.0704 |    0.0108 |                      0.1344 |                    5 |                0 |            10 |
| nested_query_selective    | matched_full_selective    | alignment_spearman |      -0.0372 |  -0.0989 |    0.0196 |                      0.1092 |                    5 |                0 |            10 |

## Nested query choices

| dimension             | query_source   |   outer_folds_selected |
|:----------------------|:---------------|-----------------------:|
| comprehensiveness     | criterion_only |                     10 |
| instruction_following | criterion_only |                      3 |
| instruction_following | matched_full   |                      7 |

## Decision

- Nested selective minus global + structure official accuracy: +0.0133 (95% CI [-0.0058, 0.0350], P(delta > 0)=0.8914, 16 net correct pairs).
- Fixed dimension policy minus global + structure: +0.0108; nested minus fixed: +0.0025.
- E1.7 official-accuracy gate: **FAIL**.
- This fail means query routing did not generalize reliably under the
  ten-question outer LOQO protocol.
- The fixed dimension policy is an exploratory, researcher-informed analysis
  motivated by E1.6; its stronger total score is not fresh confirmatory evidence.
- The nested policy records its fold-level query choices above.
  Query-policy differences should be interpreted through paired outer-fold
  accuracy rather than by selecting the highest observed point estimate.

## Leakage controls

- Query source and Ridge alpha are selected jointly inside each outer training set.
- Prospective inner validation maximizes grouped pairwise accuracy, then
  Spearman, then minimizes MAE. A retrospective note means saved predictions
  retain their historical selection objective.
- The held-out question is never used for query selection, scaling, or fitting.
- Query candidates are ordered `matched_full`, then `criterion_only`; that order
  is used only as a deterministic exact-tie break.
