# E1.1 unbounded chunk-count audit

This audit disables the per-document chunk cap while retaining the 512-token
per-chunk maximum. It covers all 160 documents (16 for each of 10 questions).

## Result

The assumption that reports for the same question have similar natural chunk
counts is not supported by this dataset.

| Question | Mean | Std | CV | Min | Max | Max/min |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 92.0 | 60.1 | 0.653 | 10 | 238 | 23.8 |
| 2 | 105.8 | 63.1 | 0.597 | 14 | 232 | 16.6 |
| 3 | 95.4 | 49.8 | 0.522 | 19 | 191 | 10.1 |
| 4 | 107.8 | 64.8 | 0.601 | 16 | 225 | 14.1 |
| 5 | 73.6 | 37.5 | 0.510 | 23 | 126 | 5.5 |
| 6 | 107.1 | 58.4 | 0.545 | 38 | 215 | 5.7 |
| 7 | 94.5 | 76.9 | 0.813 | 11 | 269 | 24.5 |
| 16 | 102.4 | 80.6 | 0.787 | 7 | 353 | 50.4 |
| 17 | 73.2 | 48.9 | 0.668 | 4 | 161 | 40.3 |
| 18 | 75.9 | 64.1 | 0.844 | 8 | 267 | 33.4 |

Across questions, the mean within-question coefficient of variation is 0.654.
Natural chunk count ranges from 4 to 353 and correlates with document token count
at 0.736. The variation therefore reflects both actual report length and document
structure, rather than only the chunking heuristic.

## Confounding diagnostics

Chunk counts are strongly associated with the generation source:

| Model | Documents | Mean chunks | Std chunks | Mean tokens |
|---|---:|---:|---:|---:|
| Gemini | 40 | 102.0 | 20.3 | 8,183 |
| Grok | 40 | 26.8 | 15.1 | 2,749 |
| Mita | 40 | 118.3 | 41.5 | 13,848 |
| Perplexity | 40 | 123.9 | 82.1 | 8,186 |

Across all documents, chunk count has raw Pearson correlations of 0.307 with
comprehensiveness, 0.283 with insight, 0.145 with instruction following, and 0.485
with readability. These are diagnostics, not valid held-out estimates, but they
show that an unnormalized variable-chunk model could exploit a length/source
shortcut.

## Decision for E1.2

The unbounded dataset is still useful because it retains all structural evidence
and contains only 14,841 chunks, which is inexpensive to embed. However, E1 should
not use raw max or fixed top-k similarity as its only rubric aggregation. It should
also include count-normalized statistics (quantiles, top-fraction means, and
normalized log-mean-exp), expose `log1p(chunk_count)` as a diagnostic control, and
compare results with the 96-chunk version under the same LOQO protocol.
