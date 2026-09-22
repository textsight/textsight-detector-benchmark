# AI-detector benchmark: scoring, per-domain false positives, and input normalisation

A reproducible harness for measuring AI-text detectors on [RAID](https://raid-bench.xyz),
and three findings it produced. Everything runs against publicly downloadable
weights, so every number here can be checked independently.

```bash
pip install -r requirements.txt
python -m benchmark.run --fetch                              # RAID labelled split, 765 MB
python -m benchmark.run --detector textsight-v23-normalised
```

The detector measured here is
[`textsightai/textsight-detector-v23-custom`](https://huggingface.co/textsightai/textsight-detector-v23-custom)
— DeBERTa-v3-large, 435M parameters, public and ungated. `model.safetensors` is
1,740,304,440 bytes, sha256
`9177c456a91e41afc6e03583d81a63a5b81d5d0a96ecfd783cd5b8c234418443`: the exact
checkpoint these numbers came from.

All figures below: 2,520 stratified RAID documents, seed 0, accuracy at a 5%
false-positive rate with a per-domain threshold.

---

## 1. Reporting a rounded probability can cost 19 points of accuracy

A confidently fine-tuned classifier saturates. Softmax returns 0.99999999, and
once that is rounded for display it becomes exactly 1.0. Two documents the model
ranked differently become indistinguishable.

| score reported | distinct values | exact ties | accuracy@5%FPR | AUROC |
|---|---|---|---|---|
| rounded probability | 578 / 2520 | 77.1% | 0.5273 | 0.8336 |
| **log-odds margin** | **2520 / 2520** | **0.0%** | **0.7144** | 0.8426 |

Paired bootstrap on the difference: **+0.187, 95% CI [+0.162, +0.196]**.

AUROC barely moves, which is the tell — the ranking was always there. What
changes is that a threshold can now separate documents rounding had glued
together. In two domains accuracy went from **zero** to 0.74 and 0.71, because
their thresholds sat at the ceiling where every document tied.

If you benchmark a detector by its reported probability, you may be measuring
its display formatting rather than its model. `HFSequenceClassifier` here
reports the margin by default.

## 2. A single false-positive rate hides a 25× spread

Fit one threshold across all human documents at 5% — the way a product ships a
single cut — then measure what that threshold actually does per domain:

| domain | FPR | | domain | FPR |
|---|---|---|---|---|
| abstracts | 0.0% | | reviews | 0.0% |
| books | 0.0% | | **wiki** | **15.3%** |
| news | 0.0% | | **recipes** | **24.7%** |
| poetry | 0.0% | | *pooled* | *5.0%* |
| reddit | 0.0% | | | |

The pooled figure is 5.0% by construction. Behind it, six domains see almost no
false positives and one sees a quarter of its human documents flagged.

This is not specific to one model — it is what happens when a detector trained
largely on prose meets formulaic, list-like text. Recipes and reference entries
are structurally close to what these models learn to call machine-written, and
human authors of such text pay for it. A headline FPR is an average over
incompatible populations.

Two caveats, both load-bearing. RAID's human text is web-scraped, so it does not
represent any particular product's users; a detector aimed at student essays
should be read off the `abstracts` and `books` rows, not the pooled figure. And
150 human documents per domain gives 5% granularity and no better.

## 3. Most cheap attacks are an input-handling problem, not a model problem

The common mechanical attacks on a text detector work the same way: leave the
text looking identical to a reader while changing the byte sequence, so the
subword tokenizer produces fragments the model never saw in training.

That is fixable before the model is involved. `NormalizingDetector` applies
NFKC, Cyrillic/Greek confusable folding, invisible-character removal, whitespace
repair and case repair, then scores. Same weights, same everything else:

| | clean accuracy | mean across 9 attacks |
|---|---|---|
| bare checkpoint | 0.7174 | 0.6087 |
| **+ input normalisation** | 0.7136 | **0.7014** |

Four of the nine attacks go to *exactly* zero effect. Clean accuracy is
unchanged within noise. The residual gap is concentrated in attacks that alter
real words rather than their encoding, which normalisation cannot repair.

The weights on HuggingFace carry no input handling — nothing does, they are just
weights — so a bare-checkpoint number measures the tokenizer's brittleness as
much as the model's judgement. Benchmark detectors in the configuration they are
deployed in, or say clearly that you did not.

Per-attack figures are printed when you run the harness. They are not tabulated
here; see [DISCLOSURE.md](DISCLOSURE.md) for why.

---

## What this measures, and what it does not

**Model-level, not product-level.** This scores a classifier plus optional input
normalisation. Commercial detectors wrap models in further signals and fusion
logic; their numbers will differ.

**Synonym and paraphrase attacks are not implemented.** Both need a model or a
thesaurus, and both are more damaging than anything included. **Every robustness
figure here is an upper bound.** A detector that survives these nine attacks is
unproven, not robust.

**RAID's test splits are blind.** `test.csv` and `test_none.csv` are
`id,generation` only, labels held by the organisers, so nothing can be scored
locally on them. Measurement uses the labelled `train_none.csv` (467,985 rows;
8 domains, 11 generators, 13,371 human documents).

**Sampling.** 150 human documents per domain, 15 per (domain, generator),
reservoir-sampled in one streaming pass, seed 0 — 2,520 documents. Human rows
get a much larger quota because they are 2.9% of the corpus and a per-domain 5%
FPR threshold cannot be fit from a handful of them.

**In-sample thresholds.** Each threshold is fit on the same human documents the
FPR is measured against, which is optimistic. RAID's own evaluation does
likewise, so figures are comparable to each other but are not held-out
estimates.

**Not a leaderboard submission.** The perturbations are independent
implementations of RAID's published attack ideas, not RAID's code, so these
numbers are not directly comparable to the RAID leaderboard.

## Adding a detector

Anything with a `name` and `score(texts) -> list[float]`, higher meaning more
likely machine-generated. The scale is irrelevant — every metric here is
rank-based or fits its own threshold.

```python
DETECTORS["my-detector"] = lambda: HFSequenceClassifier("org/model-id", "my-detector")
DETECTORS["my-detector-normalised"] = lambda: NormalizingDetector(
    HFSequenceClassifier("org/model-id", "my-detector"))
```

## Layout

```
benchmark/detectors.py   detector interface, HuggingFace adapter, normalising wrapper
benchmark/normalize.py   the input normalisation described in finding 3
benchmark/attacks.py     mechanical perturbations
benchmark/evaluate.py    per-domain threshold at target FPR, tie-aware AUROC
benchmark/run.py         CLI: sample, score, attack, report
```

## Licence

MIT. RAID is distributed by its own authors under their terms; this repository
downloads it at runtime and redistributes none of it.
