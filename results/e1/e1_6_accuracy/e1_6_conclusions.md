# E1.6 rubric attribution controls

All learned predictions use outer leave-one-question-out Ridge. Query controls
change only the rubric query text; mismatch controls use five fixed cyclic
derangements and never pair a report with its own question rubric.

## Main results

| model                              |   accuracy |   spearman |   kendall |   accuracy_comprehensiveness |   accuracy_instruction_following |    mae |
|:-----------------------------------|-----------:|-----------:|----------:|-----------------------------:|---------------------------------:|-------:|
| criterion_only_rubric              |     0.7142 |     0.5662 |    0.4283 |                       0.6717 |                           0.5217 | 0.6748 |
| matched_full_rubric_structure      |     0.7050 |     0.5424 |    0.4100 |                       0.6425 |                           0.5675 | 0.6402 |
| criterion_only_all                 |     0.7033 |     0.5662 |    0.4067 |                       0.6633 |                           0.5100 | 0.6846 |
| matched_full_all                   |     0.7008 |     0.5453 |    0.4017 |                       0.6525 |                           0.4967 | 0.6167 |
| mismatch_shift3_all                |     0.6983 |     0.5424 |    0.3967 |                       0.6483 |                           0.4317 | 0.6897 |
| generic_dimension_all              |     0.6958 |     0.5418 |    0.3917 |                       0.6558 |                           0.4117 | 0.6810 |
| mismatch_shift4_all                |     0.6925 |     0.5474 |    0.3850 |                       0.6492 |                           0.4358 | 0.7787 |
| matched_full_global_structure      |     0.6908 |     0.5226 |    0.3817 |                       0.6500 |                           0.4392 | 0.7079 |
| mismatched_ensemble_all            |     0.6892 |     0.5335 |    0.3783 |                       0.6458 |                           0.4867 | 0.7273 |
| mismatch_shift2_all                |     0.6858 |     0.5115 |    0.3717 |                       0.6317 |                           0.4267 | 0.6963 |
| matched_full_no_instruction_all    |     0.6833 |     0.5206 |    0.3667 |                       0.6500 |                           0.4775 | 0.7054 |
| mismatch_shift5_all                |     0.6800 |     0.5100 |    0.3600 |                       0.6425 |                           0.4717 | 0.7003 |
| mismatch_shift1_all                |     0.6800 |     0.4974 |    0.3600 |                       0.6192 |                           0.4358 | 0.8145 |
| prompt_only_all                    |     0.6783 |     0.4974 |    0.3567 |                       0.6183 |                           0.4450 | 0.9103 |
| matched_full_global_rubric         |     0.6783 |     0.4797 |    0.3567 |                       0.6267 |                           0.5542 | 0.6973 |
| matched_full_structure             |     0.6675 |     0.4547 |    0.3350 |                       0.6042 |                           0.4375 | 0.5958 |
| matched_full_global                |     0.6625 |     0.4594 |    0.3250 |                       0.6167 |                           0.4392 | 0.7250 |
| matched_full_rubric                |     0.6542 |     0.4097 |    0.3083 |                       0.6608 |                           0.5642 | 0.6403 |
| matched_full_no_instruction_rubric |     0.6475 |     0.4179 |    0.2950 |                       0.6400 |                           0.5567 | 0.6097 |
| mismatch_shift3_rubric             |     0.6425 |     0.4015 |    0.2850 |                       0.5492 |                           0.4442 | 0.6706 |
| prompt_only_rubric                 |     0.6333 |     0.3729 |    0.2667 |                       0.6242 |                           0.5583 | 0.6535 |
| mismatched_ensemble_rubric         |     0.6217 |     0.3515 |    0.2433 |                       0.5900 |                           0.4100 | 0.6483 |
| generic_dimension_rubric           |     0.6150 |     0.3332 |    0.2300 |                       0.5075 |                           0.4125 | 0.7888 |
| mismatch_shift1_rubric             |     0.6008 |     0.2635 |    0.2017 |                       0.5892 |                           0.4300 | 0.7401 |
| mismatch_shift2_rubric             |     0.5992 |     0.2659 |    0.1983 |                       0.5150 |                           0.3958 | 0.6687 |
| mismatch_shift5_rubric             |     0.5900 |     0.2818 |    0.1800 |                       0.5542 |                           0.4275 | 0.6613 |
| mismatch_shift4_rubric             |     0.5817 |     0.2194 |    0.1633 |                       0.5800 |                           0.4292 | 0.6658 |

## Primary official-accuracy comparisons

| candidate                  | reference                          | metric   |   mean_delta |   ci_low |   ci_high |   probability_delta_gt_zero |   positive_questions |   tied_questions |   net_correct_pairs |   n_questions |
|:---------------------------|:-----------------------------------|:---------|-------------:|---------:|----------:|----------------------------:|---------------------:|-----------------:|--------------------:|--------------:|
| matched_full_all           | matched_full_global_structure      | accuracy |       0.0100 |  -0.0092 |    0.0300 |                      0.8206 |                    6 |                0 |             12.0000 |            10 |
| criterion_only_all         | matched_full_global_structure      | accuracy |       0.0125 |  -0.0025 |    0.0325 |                      0.9146 |                    6 |                2 |             15.0000 |            10 |
| criterion_only_all         | matched_full_all                   | accuracy |       0.0025 |  -0.0100 |    0.0150 |                      0.6290 |                    4 |                2 |              3.0000 |            10 |
| matched_full_global_rubric | matched_full_global                | accuracy |       0.0158 |  -0.0075 |    0.0442 |                      0.8784 |                    7 |                0 |             19.0000 |            10 |
| matched_full_rubric        | criterion_only_rubric              | accuracy |      -0.0600 |  -0.1033 |   -0.0133 |                      0.0038 |                    2 |                0 |            -72.0000 |            10 |
| matched_full_rubric        | prompt_only_rubric                 | accuracy |       0.0208 |  -0.0350 |    0.0775 |                      0.7498 |                    6 |                0 |             25.0000 |            10 |
| matched_full_rubric        | generic_dimension_rubric           | accuracy |       0.0392 |  -0.0150 |    0.0967 |                      0.9146 |                    7 |                0 |             47.0000 |            10 |
| matched_full_rubric        | matched_full_no_instruction_rubric | accuracy |       0.0067 |  -0.0300 |    0.0442 |                      0.6256 |                    5 |                0 |              8.0000 |            10 |
| matched_full_rubric        | mismatched_ensemble_rubric         | accuracy |       0.0325 |   0.0041 |    0.0608 |                      0.9864 |                    8 |                0 |             39.0000 |            10 |
| matched_full_all           | generic_dimension_all              | accuracy |       0.0050 |  -0.0158 |    0.0250 |                      0.6756 |                    6 |                0 |              6.0000 |            10 |
| matched_full_all           | mismatched_ensemble_all            | accuracy |       0.0117 |  -0.0067 |    0.0317 |                      0.8854 |                    5 |                2 |             14.0000 |            10 |

## Dimension-mechanism diagnostics

| candidate                  | reference                          | metric             |   mean_delta |   ci_low |   ci_high |   probability_delta_gt_zero |   positive_questions |   tied_questions |   n_questions |
|:---------------------------|:-----------------------------------|:-------------------|-------------:|---------:|----------:|----------------------------:|---------------------:|-----------------:|--------------:|
| matched_full_all           | matched_full_global_structure      | alignment_accuracy |       0.0300 |  -0.0192 |    0.0871 |                      0.8700 |                    6 |                0 |            10 |
| matched_full_all           | matched_full_global_structure      | alignment_spearman |       0.0726 |  -0.0241 |    0.1760 |                      0.9294 |                    7 |                0 |            10 |
| criterion_only_all         | matched_full_global_structure      | alignment_accuracy |       0.0421 |  -0.0108 |    0.1133 |                      0.9246 |                    6 |                0 |            10 |
| criterion_only_all         | matched_full_global_structure      | alignment_spearman |       0.0742 |   0.0002 |    0.1534 |                      0.9754 |                    7 |                0 |            10 |
| criterion_only_all         | matched_full_all                   | alignment_accuracy |       0.0121 |  -0.0558 |    0.0888 |                      0.6266 |                    5 |                1 |            10 |
| criterion_only_all         | matched_full_all                   | alignment_spearman |       0.0015 |  -0.0732 |    0.0801 |                      0.5062 |                    5 |                0 |            10 |
| matched_full_global_rubric | matched_full_global                | alignment_accuracy |       0.0625 |  -0.0112 |    0.1483 |                      0.9488 |                    7 |                0 |            10 |
| matched_full_global_rubric | matched_full_global                | alignment_spearman |       0.0904 |  -0.0161 |    0.2044 |                      0.9520 |                    6 |                0 |            10 |
| matched_full_rubric        | criterion_only_rubric              | alignment_accuracy |       0.0158 |  -0.0617 |    0.0896 |                      0.6488 |                    5 |                0 |            10 |
| matched_full_rubric        | criterion_only_rubric              | alignment_spearman |       0.0745 |  -0.1133 |    0.2562 |                      0.7790 |                    5 |                0 |            10 |
| matched_full_rubric        | prompt_only_rubric                 | alignment_accuracy |       0.0212 |  -0.0329 |    0.0721 |                      0.7944 |                    5 |                0 |            10 |
| matched_full_rubric        | prompt_only_rubric                 | alignment_spearman |       0.0829 |  -0.0617 |    0.2156 |                      0.8840 |                    7 |                0 |            10 |
| matched_full_rubric        | generic_dimension_rubric           | alignment_accuracy |       0.1525 |   0.0942 |    0.2146 |                      1.0000 |                   10 |                0 |            10 |
| matched_full_rubric        | generic_dimension_rubric           | alignment_spearman |       0.3776 |   0.2498 |    0.5163 |                      1.0000 |                   10 |                0 |            10 |
| matched_full_rubric        | matched_full_no_instruction_rubric | alignment_accuracy |       0.0142 |  -0.0158 |    0.0437 |                      0.8164 |                    6 |                0 |            10 |
| matched_full_rubric        | matched_full_no_instruction_rubric | alignment_spearman |       0.0452 |  -0.0113 |    0.0997 |                      0.9440 |                    7 |                0 |            10 |
| matched_full_rubric        | mismatched_ensemble_rubric         | alignment_accuracy |       0.1125 |   0.0496 |    0.1750 |                      1.0000 |                    8 |                0 |            10 |
| matched_full_rubric        | mismatched_ensemble_rubric         | alignment_spearman |       0.3199 |   0.1617 |    0.4630 |                      0.9998 |                    9 |                0 |            10 |
| matched_full_all           | generic_dimension_all              | alignment_accuracy |       0.0408 |  -0.0113 |    0.1025 |                      0.9200 |                    5 |                0 |            10 |
| matched_full_all           | generic_dimension_all              | alignment_spearman |       0.0559 |  -0.0262 |    0.1514 |                      0.8892 |                    6 |                0 |            10 |
| matched_full_all           | mismatched_ensemble_all            | alignment_accuracy |       0.0083 |  -0.0250 |    0.0383 |                      0.6976 |                    6 |                1 |            10 |
| matched_full_all           | mismatched_ensemble_all            | alignment_spearman |       0.0393 |  -0.0335 |    0.1066 |                      0.8592 |                    7 |                0 |            10 |

## Decision

- Matched full features add +0.0100 official
  pairwise accuracy over `global+structure` (95% question-bootstrap CI [-0.0092, 0.0300]; 12 net correct pairs).
- `criterion_only_all` adds +0.0125 accuracy over global+structure
  (15 net correct pairs; P(delta>0)=0.9146) and is
  the E1.6 accuracy leader; full prompt context is not uniformly useful.
- Matched rubric beats the mismatched ensemble by +0.0325
  accuracy and generic dimensions by +0.0392.
- Dimension accuracy and Spearman remain mechanism diagnostics; they do not
  replace the official accuracy of the weighted total.
- E1.6 supports selective fixed-representation rubric routing. Whether to train
  a learned interaction is a separate supervision and generalization question.

## Interpretation rules

- `matched_full_all > matched_full_global_structure` tests incremental rubric value.
- Matched versus generic or mismatched tests criterion-specific conditioning.
- Official accuracy is total correct weighted-score pairs divided by total pairs.
- Spearman diagnoses large rank displacement; Kendall is mainly an accuracy
  consistency check when weighted totals have no ties.
- `mismatched_ensemble_*` averages predictions from five independently retrained
  mismatch controls and is therefore a conservative negative control.
- The held-out question is never used for scaling, alpha selection, or fitting.
