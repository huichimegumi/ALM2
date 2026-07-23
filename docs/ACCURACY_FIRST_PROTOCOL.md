# Accuracy-first evaluation protocol

From the next experiment onward, the primary endpoint is the exact official
pairwise accuracy of the rubric-weighted total score:

```text
sum(correct document pairs across questions)
------------------------------------------------
sum(all evaluated document pairs across questions)
```

The uncertainty unit is the question. A paired bootstrap resamples questions
and recomputes pooled correct pairs divided by pooled total pairs. It does not
treat document pairs as independent observations. Every comparison reports the
accuracy delta, 95% question-bootstrap interval, bootstrap probability of a
positive delta, positive/tied question counts, and net correct document pairs.

Spearman is secondary and diagnoses large rank displacement that pairwise
accuracy may underweight. Kendall is retained as a consistency check; on the
current weighted totals, which have almost no ties, it is nearly a deterministic
transform of accuracy. Dimension-level accuracy is a mechanism diagnostic and
does not replace accuracy of the official weighted total.

For grouped inner selection, candidate hyperparameters and query
representations are ordered lexicographically:

1. maximize within-question pairwise accuracy pooled over validation questions;
2. maximize macro within-question Spearman;
3. minimize document-level MAE;
4. use the declared candidate order only for an exact tie.

Outer held-out questions remain unavailable to preprocessing, selection,
fitting, and early stopping.

## Retrospective status

E0 through E2-A0.1 were originally designed or selected after inspecting
Spearman and, for Ridge models, often used inner MAE. Their saved predictions
are retained unchanged. Accuracy-first reinterpretation of those predictions is
exploratory and must not be described as fresh confirmatory evidence. Running
the updated pipelines creates a new accuracy-selected experiment and should use
a new output directory rather than overwrite historical predictions.
