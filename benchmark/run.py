#!/usr/bin/env python3
"""Reproduce the published numbers, or measure your own detector.

    python -m benchmark.run --fetch                      # download RAID (765 MB)
    python -m benchmark.run --detector textsight-v23     # clean + all attacks

Everything is stratified from RAID's labelled `train_none.csv`. RAID's *test*
splits are `id,generation` only - deliberately blind - so no local evaluation is
possible on them; that is why measurement happens on the training split.

Two robustness numbers are reported per attack, because they answer different
questions:

  evasion TPR   threshold fit on CLEAN human text, TPR measured on ATTACKED
                machine text. The security question: can someone perturb
                machine text past a detector calibrated on ordinary writing?

  RAID-like     human text attacked too, threshold refit on it. Closer to what
                a leaderboard computes, and more forgiving, because perturbing
                human text raises the threshold as well.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import sys
import time
import urllib.request

from .attacks import ATTACKS
from .detectors import build
from .evaluate import auroc

csv.field_size_limit(10_000_000)

RAID_URL = "https://dataset.raid-bench.xyz/train_none.csv"
DEFAULT_CSV = os.path.join("data", "train_none.csv")


def fetch(dest: str = DEFAULT_CSV):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest):
        print(f"already present: {dest} ({os.path.getsize(dest)/1048576:.0f} MB)")
        return
    print(f"downloading {RAID_URL} -> {dest} (765 MB)")

    def hook(done, block, total):
        got = done * block
        sys.stdout.write(f"\r  {got/1048576:8.1f} / {total/1048576:.0f} MB")
        sys.stdout.flush()

    tmp = dest + ".part"
    urllib.request.urlretrieve(RAID_URL, tmp, hook)
    os.replace(tmp, dest)
    print("\ndone")


def sample(path: str, human_per_domain: int, ai_per_group: int, seed: int):
    """Reservoir-sample in one streaming pass.

    The file is 468k rows / 765 MB; materialising it as dicts costs several GB.
    Human rows get their own, much larger quota because they are only 2.9% of
    the corpus and a per-domain threshold at 5% FPR cannot be fit from a
    handful of them - 100 humans gives 5% granularity, and no better.
    """
    rnd = random.Random(seed)
    buckets: dict = collections.defaultdict(list)
    counts: dict = collections.defaultdict(int)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            is_human = r.get("model") == "human"
            key = ("human", r.get("domain")) if is_human else \
                  ("ai", r.get("domain"), r.get("model"))
            cap = human_per_domain if is_human else ai_per_group
            counts[key] += 1
            b = buckets[key]
            if len(b) < cap:
                b.append(r)
            else:
                j = rnd.randrange(counts[key])
                if j < cap:
                    b[j] = r
    docs = [{"domain": r["domain"], "model": r["model"],
             "text": (r["generation"] or "").strip()}
            for k in sorted(buckets, key=str) for r in buckets[k]]
    rnd.shuffle(docs)
    h = sum(1 for d in docs if d["model"] == "human")
    print(f"sample: {h} human + {len(docs)-h} machine = {len(docs)} documents, "
          f"{len(set(d['domain'] for d in docs))} domains")
    return docs


def thr_at_fpr(human: list[float], target: float = 0.05):
    if not human:
        return None
    for x in sorted(set(human)):
        if sum(1 for s in human if s >= x) / len(human) <= target:
            return x
    return max(human) + 1e-9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true", help="download RAID and exit")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--detector", default="textsight-v23")
    p.add_argument("--human-per-domain", type=int, default=150)
    p.add_argument("--ai-per-group", type=int, default=15)
    p.add_argument("--rate", type=float, default=0.15, help="attack strength")
    p.add_argument("--fpr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join("runs", "results.json"))
    a = p.parse_args()

    if a.fetch:
        fetch(a.csv)
        return
    if not os.path.exists(a.csv):
        raise SystemExit(f"{a.csv} not found - run with --fetch first")

    det = build(a.detector)
    docs = sample(a.csv, a.human_per_domain, a.ai_per_group, a.seed)

    t0 = time.time()
    for d, s in zip(docs, det.score([d["text"] for d in docs])):
        d["clean"] = s
    print(f"clean scored in {time.time()-t0:.0f}s\n")

    clean_thr = {}
    per_dom = collections.defaultdict(list)
    for d in docs:
        if d["model"] == "human":
            per_dom[d["domain"]].append(d["clean"])
    for dom, v in per_dom.items():
        clean_thr[dom] = thr_at_fpr(v, a.fpr)

    def summarise(key):
        ev_hit = ev_n = 0
        rl = collections.defaultdict(lambda: {"h": [], "a": []})
        ys, ss = [], []
        for d in docs:
            s, is_h = d[key], d["model"] == "human"
            rl[d["domain"]]["h" if is_h else "a"].append(s)
            ys.append(0 if is_h else 1)
            ss.append(s)
            if not is_h and clean_thr.get(d["domain"]) is not None:
                ev_n += 1
                ev_hit += s >= clean_thr[d["domain"]]
        tp = n = 0
        for dom, x in rl.items():
            if not x["h"] or not x["a"]:
                continue
            t = thr_at_fpr(x["h"], a.fpr)
            tp += sum(1 for s in x["a"] if s >= t)
            n += len(x["a"])
        return ev_hit / max(1, ev_n), tp / max(1, n), auroc(ys, ss)

    results = {}
    base = summarise("clean")
    results["none"] = dict(zip(("evasion_tpr", "raid_like", "auroc"), base))
    print(f"detector: {det.name}   attack rate: {a.rate}   FPR target: {a.fpr:.0%}\n")
    print(f"{'attack':<22}{'evasion TPR':>13}{'RAID-like':>11}{'AUROC':>9}")
    print(f"{'none (baseline)':<22}{base[0]:>13.4f}{base[1]:>11.4f}{base[2]:>9.4f}")

    for name, fn in ATTACKS.items():
        rnd = random.Random(a.seed)
        for d, s in zip(docs, det.score([fn(d["text"], a.rate, rnd) for d in docs])):
            d[name] = s
        ev, rl_, au = summarise(name)
        results[name] = {"evasion_tpr": ev, "raid_like": rl_, "auroc": au}
        print(f"{name:<22}{ev:>13.4f}{rl_:>11.4f}{au:>9.4f}   "
              f"{ev-base[0]:+.4f} / {rl_-base[1]:+.4f}")

    # What a SINGLE headline false-positive rate hides.
    #
    # Fit one threshold across all human documents at the target FPR - the way
    # a product ships a single cut - then measure the false-positive rate that
    # threshold actually produces in each domain. The pooled figure is by
    # construction the target; the per-domain spread is the honest picture, and
    # it is exactly what a single headline number conceals.
    all_human = [d["clean"] for d in docs if d["model"] == "human"]
    g = thr_at_fpr(all_human, a.fpr)
    print(f"\nfalse positives on CLEAN human text at ONE pooled threshold "
          f"({g:.3f}, fit to {a.fpr:.0%} overall):")
    rates = {}
    for dom in sorted(per_dom):
        v = per_dom[dom]
        rates[dom] = sum(1 for s in v if s >= g) / len(v)
        print(f"  {dom:<12} n={len(v):<5} FPR {rates[dom]:>7.1%}")
    pooled = sum(1 for s in all_human if s >= g) / len(all_human)
    print(f"  {'POOLED':<12} n={len(all_human):<5} FPR {pooled:>7.1%}"
          f"   <- the single number")
    print(f"  per-domain spread: {min(rates.values()):.1%} to {max(rates.values()):.1%}")

    worst = min(((k, v) for k, v in results.items() if k != "none"),
                key=lambda kv: kv[1]["evasion_tpr"])
    mean_rl = sum(v["raid_like"] for k, v in results.items() if k != "none") \
        / (len(results) - 1)
    print(f"\nworst attack: {worst[0]} (evasion TPR {worst[1]['evasion_tpr']:.4f}, "
          f"{worst[1]['evasion_tpr']-base[0]:+.4f})")
    print(f"mean RAID-like across {len(results)-1} attacks: {mean_rl:.4f} "
          f"(clean {base[1]:.4f})")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"detector": det.name, "rate": a.rate, "fpr": a.fpr,
               "seed": a.seed, "n_docs": len(docs), "results": results,
               "pooled_threshold": g, "fpr_by_domain": rates},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("NOTE: synonym and paraphrase attacks are NOT implemented here. They "
          "need a model or thesaurus and are the two most likely to hurt, so "
          "these numbers are an upper bound on adversarial robustness.")


if __name__ == "__main__":
    main()
