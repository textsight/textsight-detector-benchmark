"""Metrics.

`threshold_at_fpr` fits the lowest threshold whose false-positive rate on human
documents is at or below the target - per domain, because detector behaviour
varies enormously across text types and a single pooled threshold hides that.

`auroc` is rank-based with correct handling of TIES, which matters here: a
saturated detector puts most of its documents on a handful of values, and a
tie-naive implementation will quietly flatter it.
"""
import argparse, json
from collections import defaultdict


def auroc(ys, ss):
    """Rank-based AUROC with correct handling of ties (ties are the whole point here)."""
    pairs = sorted(zip(ss, ys))
    ranks, i, n = [0.0] * len(pairs), 0, len(pairs)
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(1 for _, y in pairs if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    s = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (s - pos * (pos + 1) / 2.0) / (pos * neg)


def threshold_at_fpr(human_scores, target):
    """Lowest threshold whose FPR on human docs is <= target."""
    if not human_scores:
        return None
    for t in sorted(set(human_scores)):
        if sum(1 for s in human_scores if s >= t) / len(human_scores) <= target:
            return t
    return max(human_scores) + 1e-9


def report(recs, field, target):
    by_dom = defaultdict(lambda: {"human": [], "ai": []})
    for r in recs:
        v = r.get(field)
        if v is None:
            continue
        by_dom[r.get("domain")]["human" if r.get("model") == "human" else "ai"].append(v)

    tp = n = 0
    ys, ss = [], []
    for dom, d in sorted(by_dom.items(), key=lambda kv: str(kv[0])):
        if not d["human"] or not d["ai"]:
            print(f"    {str(dom):<12} SKIPPED (human={len(d['human'])}, ai={len(d['ai'])})")
            continue
        t = threshold_at_fpr(d["human"], target)
        hit = sum(1 for s in d["ai"] if s >= t)
        tp += hit; n += len(d["ai"])
        ys += [0] * len(d["human"]) + [1] * len(d["ai"])
        ss += d["human"] + d["ai"]
        print(f"    {str(dom):<12} thr={t:7.2f}  TPR={hit/len(d['ai']):.4f}  "
              f"(ai={len(d['ai'])}, human={len(d['human'])})")
    print(f"  == {field}: accuracy@{target:.0%}FPR = {tp/max(1,n):.4f}   "
          f"AUROC = {auroc(ys, ss):.4f}   n={n}")


# The CLI lives in run.py; this module is the metric library.
