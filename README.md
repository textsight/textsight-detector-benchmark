# AI-detector benchmark: scoring, per-domain false positives, and input normalisation

A reproducible harness for measuring AI-text detectors on [RAID](https://raid-bench.xyz),
a leaderboard of nine open detectors, and what building it revealed about how
detector comparisons go wrong. Everything runs against publicly downloadable
weights, so every number here can be checked independently.

```bash
pip install -r requirements.txt
python -m benchmark.run --fetch                              # RAID labelled split, 765 MB
python -m benchmark.run --detector textsight-v23-normalised
```

Findings 1-3 use
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

Normalisation defends attacks on the *encoding* of text. It does nothing against
attacks that change the words themselves - synonym substitution and misspelling
pass straight through it, and are among the strongest measured here. That split
is the useful generalisation: input hygiene is cheap and closes a whole class,
and the class it cannot close is the one that needs a better model.

Per-attack figures are printed when you run the harness. They are not tabulated
here; see [DISCLOSURE.md](DISCLOSURE.md) for why.

## 3b. Paraphrasing machine text with another model makes it MORE detectable

The obvious way to evade a detector is to rewrite the text. Measured against a
public sentence-level paraphraser
([`humarin/chatgpt_paraphraser_on_T5_base`](https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base)),
rewriting a rising fraction of each document's sentences:

| fraction of sentences rewritten | evasion TPR | vs clean |
|---|---|---|
| 0 (baseline) | 0.7136 | — |
| 0.15 | 0.7068 | −0.007 |
| 0.50 | 0.7045 | −0.009 |
| **1.00** | **0.7318** | **+0.018** |

Rewriting *everything* leaves the text **easier** to detect than leaving it
alone. The reason is not mysterious once stated: the paraphraser is itself a
language model, so its output carries its own machine-generated signature.
Full paraphrasing does not launder a machine fingerprint, it replaces one with a
fresher one.

This matters because "just paraphrase it" is the assumed defeat condition for
detectors, and for a small seq2seq model it is not one. It also means a
dose-response curve is worth measuring: a single point at a low rate would have
reported this attack as mildly effective, and a single point at full rate as
mildly helpful, and neither alone is the finding.

**What this does not show.** One paraphraser, sentence-level. RAID's own
paraphrase attack uses DIPPER, a stronger paragraph-level model, and an
instruction-tuned LLM asked to rewrite in a human register is untested here and
is a different proposition entirely. Paraphrase is not solved - this particular
paraphraser just is not the way to do it.

---

## 4. Two ways a detector comparison silently breaks

Running nine detectors through one harness turned up two failure modes that
would quietly invalidate any comparison built without checking for them. Both
are handled here automatically, and both are worth knowing about if you build
your own.

**Three of nine had reversed or undocumented label polarity.** `radar`,
`openai-roberta-large` and `piratexx` rank human text *above* machine text under
a naive reading of their label maps. Several models on the Hub expose bare
`LABEL_0`/`LABEL_1` with no documented direction. Get it backwards and you invert
a detector completely, reporting a strong one as far worse than random. Direction
is therefore established empirically on a small labelled probe, not read off the
config - see `detectors.orient`.

**One of nine was trained on the benchmark.** RAID's test splits are blind, so
measurement has to happen on its labelled `train_none` split - and a detector
fine-tuned on RAID is then being evaluated on its own training data.
`MayZhou/e5-small-lora-ai-generated-detector` declares `datasets: [liamdugan/raid]`
and posts the highest score in the field by a wide margin, which is what
in-domain evaluation looks like. `benchmark/contamination.py` reads each model's
declared datasets and sorts results into three tiers so an in-domain score cannot
sit at the top of a table of out-of-domain ones.

Declared metadata is not proof. A model can omit its training data or describe it
only in prose, so "nothing declared" means exactly that - not "verified clean".

## 5. Leaderboard

2,520 documents, seed 0, accuracy at 5% FPR with per-domain thresholds, scored on
the log-odds margin, auto-oriented. `worst FPR` is the worst single domain's
false-positive rate at one pooled 5% threshold - what a headline number hides.

**No RAID overlap declared.** Status `undeclared` means the model publishes no
dataset list, so contamination is unverified in either direction.

| detector | acc@5%FPR | AUROC | worst FPR | contamination |
|---|---|---|---|---|
| textsight-v23 | **0.7174** | 0.8436 | 24.7% | undeclared |
| textsight-v23-normalised | 0.7136 | 0.8422 | 24.7% | undeclared |
| radar | 0.6712 | **0.8927** | 18.7% | undeclared |
| piratexx | 0.6348 | 0.8176 | 18.7% | undeclared |
| hc3-roberta | 0.4485 | 0.7280 | 22.0% | none declared |

**Training data overlaps a RAID domain — read with care.**

| detector | acc@5%FPR | AUROC | worst FPR | overlap |
|---|---|---|---|---|
| roberta-mixed | 0.8818 | 0.9397 | 12.7% | arXiv abstracts → `abstracts` |
| openai-roberta-base | 0.6068 | 0.8439 | 13.3% | Wikipedia → `wiki` |
| openai-roberta-large | 0.5879 | 0.8322 | 12.0% | Wikipedia → `wiki` |

**Trained on RAID — not comparable.**

| detector | acc@5%FPR | AUROC | worst FPR | |
|---|---|---|---|---|
| e5-small-lora | 0.9364 | 0.9867 | 9.3% | in-domain |

Three things in that table are worth more than the ordering:

**AUROC and threshold accuracy disagree.** `radar` has the best AUROC of any
uncontaminated model (0.8927) and ranks below `textsight-v23` at the 5%
operating point. Ranking quality and threshold behaviour are different
properties, and a benchmark that reports only one of them will mislead you.

**Every detector has a domain where it is far worse than its headline.** The
worst-domain false-positive rate ranges from 9.3% to 24.7% against a 5% pooled
target. Not one of the nine is uniform across text types.

**`textsight-v23` has the highest worst-domain FPR in the field.** It leads on
accuracy and is the most likely of these nine to falsely flag a human author in
its weakest genre. Those are both true, and a comparison that reported only the
first would be the kind of thing this repository exists to argue against.

Reproduce with:

```bash
python -m benchmark.leaderboard                 # all detectors
python -m benchmark.leaderboard --attacks       # also per-attack evasion
```

---

## What this measures, and what it does not

**Model-level, not product-level.** This scores a classifier plus optional input
normalisation. Commercial detectors wrap models in further signals and fusion
logic; their numbers will differ.

**Only open models are here.** Closed commercial detectors are absent because
they cannot be measured reproducibly through an API that may change under you.
That is a property of those products rather than an omission, but it does mean
this leaderboard is not a market survey.

**One entry is our own.** `textsight-v23` is published by the same account as
this repository. That is a reason to check the numbers rather than take them -
which is the entire point of shipping the code, the seed, and the checkpoint
hash. It is also why the section above says plainly that our model has the worst
worst-domain false-positive rate of the nine.

**Eleven attacks, not the full space.** Synonym substitution and paraphrase are
now measured (findings 3 and 3b); `synonym` needs WordNet and `paraphrase` needs
a seq2seq model, and both are skipped with a warning if the dependency is absent.
What remains untested is the strongest form of paraphrase - a paragraph-level
model like DIPPER, or an instruction-tuned LLM told to rewrite in a human
register. Robustness figures here are still an upper bound, just a tighter one
than before.

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

## Preprint

A write-up of these findings is in [`paper/`](paper/) as LaTeX source. It is a
measurement paper about evaluation artifacts rather than a claim that any detector
is good: each finding is of the form "this factor moved the headline metric by
more than the differences normally used to rank detectors against each other".

## Licence

MIT. RAID is distributed by its own authors under their terms; this repository
downloads it at runtime and redistributes none of it.
