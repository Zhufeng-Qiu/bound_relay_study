# B1 — the compression ratio is predictable from free metadata

3456 observations (1152 tensors × 3 error bounds), 24 documents, cuSZp `fixed`.
GroupKFold by **document**; model family and hyperparameters chosen by an inner
grouped CV inside training folds only. Target is log ratio. Local, **$0**.
Raw: `results/public/b1_predictability/predictability.json`.

| feature set | ratio MAE | 95% CI (document bootstrap) | byte MAPE |
|---|---|---|---|
| constant (predict the mean) | 0.4288× | — | — |
| **metadata-only** | **0.0250×** | [0.0245, 0.0255] | **1.01%** |
| statistics-aware | 0.0222× | [0.0213, 0.0232] | 0.93% |

`metadata` is layer depth, K/V, sequence length and ε — all known from the request
before anything is compressed. `statistics` adds a pass over the tensor: std,
range, outlier rate, near-zero fraction, adjacent-difference spread.

**Metadata alone removes 94.2% of the error a constant baseline leaves, and
predicts the compressed size to within 1%.** Reading the tensor buys a further
11%, which does not pay for a full pass over the data it requires.

This matters because of what the decision needs. A transport controller must
choose compress or bypass *before* the compressed size exists; a predictor that
needs the data has already paid most of the cost it was meant to avoid. Here the
free features are almost as good as the paid ones.

## The task is easier than it looks, and that is a finding too

cuSZp's ratio on this corpus spans only 1.83–3.58× (sd 0.488). B0 already found it
flat — roughly 10% spread across every layer, both kinds and all four lengths — so
predicting it accurately is not the same achievement it would be on a corpus with
real variance. The honest statement is that **on this model and corpus the ratio
is nearly a function of the request metadata**, not that ratio prediction is
solved in general.

## Limits

* One model, one corpus, one codec mode. Whether the near-determinism survives a
  different model, a different compressor, or a workload with mixed sequence
  lengths is untested.
* Splits are grouped by document, and the CI is a document-level bootstrap over
  24 clusters — wide enough that small differences between feature sets should not
  be over-read.
* Latency is deliberately not modelled here. B0 showed warm medians flat at
  ~0.88 ms across every mode and tensor, so the cost side uses measured medians
  and an interval rather than a fitted predictor.
