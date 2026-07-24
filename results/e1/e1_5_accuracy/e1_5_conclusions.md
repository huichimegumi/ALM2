# E1.5 Huber and within-question pairwise results

All MLP predictions are outer leave-one-question-out and averaged over three fixed seeds.
Training epochs, architecture, pair weight, and preprocessing are fixed before evaluation;
held-out question labels are never used for early stopping or model selection.

## Main results

| model                    |   accuracy |   spearman |   kendall |    mae |   rmse |
|:-------------------------|-----------:|-----------:|----------:|-------:|-------:|
| ridge_all_unbounded      |     0.7008 |     0.5453 |    0.4017 | 0.6167 | 0.7647 |
| mlp_all_huber_pair       |     0.6792 |     0.4782 |    0.3583 | 0.7108 | 0.8653 |
| mlp_all_huber            |     0.6742 |     0.4647 |    0.3483 | 0.6781 | 0.8280 |
| mlp_rubric_huber_pair    |     0.6583 |     0.4209 |    0.3167 | 0.6405 | 0.8088 |
| mlp_rubric_huber         |     0.6500 |     0.4065 |    0.3000 | 0.6260 | 0.7823 |
| mlp_structure_huber_pair |     0.6458 |     0.4050 |    0.2917 | 0.6493 | 0.8494 |
| mlp_structure_huber      |     0.6325 |     0.3791 |    0.2650 | 0.6456 | 0.8371 |

## Pairwise-loss checks

- structure: pairwise minus Huber accuracy = +0.0133; Spearman = +0.0259.
- rubric: pairwise minus Huber accuracy = +0.0083; Spearman = +0.0144.
- all: pairwise minus Huber accuracy = +0.0050; Spearman = +0.0135.

## Paired question bootstrap

| candidate                | reference           | metric   |   mean_delta |   ci_low |   ci_high |   probability_delta_gt_zero |   positive_questions |   tied_questions |   net_correct_pairs |   n_questions |
|:-------------------------|:--------------------|:---------|-------------:|---------:|----------:|----------------------------:|---------------------:|-----------------:|--------------------:|--------------:|
| mlp_structure_huber_pair | mlp_structure_huber | accuracy |       0.0133 |  -0.0008 |    0.0267 |                      0.9624 |                    6 |                3 |             16.0000 |            10 |
| mlp_structure_huber_pair | mlp_structure_huber | spearman |       0.0259 |  -0.0103 |    0.0591 |                      0.9174 |                    7 |                0 |            nan      |            10 |
| mlp_structure_huber_pair | mlp_structure_huber | kendall  |       0.0267 |  -0.0017 |    0.0533 |                      0.9730 |                    6 |                3 |            nan      |            10 |
| mlp_rubric_huber_pair    | mlp_rubric_huber    | accuracy |       0.0083 |  -0.0075 |    0.0333 |                      0.7052 |                    3 |                5 |             10.0000 |            10 |
| mlp_rubric_huber_pair    | mlp_rubric_huber    | spearman |       0.0144 |  -0.0238 |    0.0709 |                      0.6734 |                    4 |                1 |            nan      |            10 |
| mlp_rubric_huber_pair    | mlp_rubric_huber    | kendall  |       0.0167 |  -0.0150 |    0.0650 |                      0.7138 |                    3 |                5 |            nan      |            10 |
| mlp_all_huber_pair       | mlp_all_huber       | accuracy |       0.0050 |  -0.0033 |    0.0142 |                      0.8468 |                    4 |                3 |              6.0000 |            10 |
| mlp_all_huber_pair       | mlp_all_huber       | spearman |       0.0135 |  -0.0115 |    0.0344 |                      0.8688 |                    6 |                1 |            nan      |            10 |
| mlp_all_huber_pair       | mlp_all_huber       | kendall  |       0.0100 |  -0.0067 |    0.0283 |                      0.8886 |                    4 |                3 |            nan      |            10 |
| mlp_all_huber            | ridge_all_unbounded | accuracy |      -0.0267 |  -0.0525 |   -0.0042 |                      0.0084 |                    2 |                0 |            -32.0000 |            10 |
| mlp_all_huber            | ridge_all_unbounded | spearman |      -0.0806 |  -0.1462 |   -0.0309 |                      0.0000 |                    1 |                0 |            nan      |            10 |
| mlp_all_huber            | ridge_all_unbounded | kendall  |      -0.0533 |  -0.1033 |   -0.0083 |                      0.0078 |                    2 |                0 |            nan      |            10 |
| mlp_all_huber_pair       | ridge_all_unbounded | accuracy |      -0.0217 |  -0.0442 |   -0.0008 |                      0.0212 |                    3 |                1 |            -26.0000 |            10 |
| mlp_all_huber_pair       | ridge_all_unbounded | spearman |      -0.0671 |  -0.1224 |   -0.0153 |                      0.0042 |                    3 |                0 |            nan      |            10 |
| mlp_all_huber_pair       | ridge_all_unbounded | kendall  |      -0.0433 |  -0.0883 |   -0.0017 |                      0.0216 |                    3 |                1 |            nan      |            10 |

## Interpretation rules

- Pairwise improvements must be judged against the same architecture and features.
- A similar gain for structure and rubric indicates a generic ranking-objective effect.
- Huber degradation with ranking improvement is an expected calibration/ranking tradeoff.
- With 10 questions, paired question-bootstrap intervals remain the uncertainty unit.
