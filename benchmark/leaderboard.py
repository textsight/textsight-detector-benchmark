#!/usr/bin/env python3
"""Compare several detectors on the same documents, under the same rules.

    python -m benchmark.leaderboard                       # all registered detectors
    python -m benchmark.leaderboard --detectors radar,hc3-roberta
    python -m benchmark.leaderboard --attacks              # also run perturbations

Every detector sees exactly the same sample, the same thresholds fitted the same
way, and the same metric. Each is auto-oriented on a labelled probe first, so a
detector whose label map is undocumented or reversed is not unfairly sunk by a
polarity mistake - see detectors.orient.

Only open models with standard architectures are included. Commercial detectors
are absent because they are closed APIs and cannot be measured reproducibly;
that is a property of those products, not an omission here.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time

from .attacks import ATTACKS
from .contamination import BADGE, check
from .detectors import DETECTORS, OrientedDetector, build, orient
from .evaluate import auroc
from .run import DEFAULT_CSV, sample, thr_at_fpr


def measure(det, docs, fpr, attacks: bool, rate: float, seed: int):
    """Score one detector on clean text, and optionally under each attack."""
    out = {}
    clean = det.score([d["text"] for d in docs])
    for d, s in zip(docs, clean):
        d["_s"] = s

    per_dom = collections.defaultdict(list)
    for d in docs:
        if d["model"] == "human":
            per_dom[d["domain"]].append(d["_s"])
    clean_thr = {k: thr_at_fpr(v, fpr) for k, v in per_dom.items()}

    def acc(key):
        rl = collections.defaultdict(lambda: {"h": [], "a": []})
        ys, ss = [], []
        for d in docs:
            v = d[key]
            rl[d["domain"]]["h" if d["model"] == "human" else "a"].append(v)
            ys.append(0 if d["model"] == "human" else 1)
            ss.append(v)
        tp = n = 0
        for dom, x in rl.items():
            if not x["h"] or not x["a"]:
                continue
            t = thr_at_fpr(x["h"], fpr)
            tp += sum(1 for s in x["a"] if s >= t)
            n += len(x["a"])
        return tp / max(1, n), auroc(ys, ss)

    out["clean"], out["auroc"] = acc("_s")

    # Saturation: how much of the score range the detector actually uses. A
    # detector that pins most documents on a few values cannot be thresholded,
    # however good its underlying ranking is.
    vals = [d["_s"] for d in docs]
    out["distinct_frac"] = len(set(vals)) / len(vals)

    # Worst-domain false positives at one pooled threshold - what a single
    # headline FPR would hide for this detector.
    all_h = [d["_s"] for d in docs if d["model"] == "human"]
    g = thr_at_fpr(all_h, fpr)
    dom_fpr = {dom: sum(1 for s in v if s >= g) / len(v) for dom, v in per_dom.items()}
    out["worst_domain_fpr"] = max(dom_fpr.values())
    out["worst_domain"] = max(dom_fpr, key=dom_fpr.get)

    if attacks:
        ev = {}
        for name, fn in ATTACKS.items():
            rnd = random.Random(seed)
            for d, s in zip(docs, det.score([fn(d["text"], rate, rnd) for d in docs])):
                d[name] = s
            hit = n = 0
            for d in docs:
                if d["model"] == "human":
                    continue
                t = clean_thr.get(d["domain"])
                if t is not None:
                    n += 1
                    hit += d[name] >= t
            ev[name] = hit / max(1, n)
        out["evasion"] = ev
        out["attack_mean"] = sum(ev.values()) / len(ev)
        out["attack_worst"] = min(ev.values())
        out["attack_worst_name"] = min(ev, key=ev.get)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--detectors", default="")
    p.add_argument("--human-per-domain", type=int, default=150)
    p.add_argument("--ai-per-group", type=int, default=15)
    p.add_argument("--fpr", type=float, default=0.05)
    p.add_argument("--rate", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attacks", action="store_true")
    p.add_argument("--out", default=os.path.join("runs", "leaderboard.json"))
    a = p.parse_args()

    if not os.path.exists(a.csv):
        raise SystemExit(f"{a.csv} not found - run `python -m benchmark.run --fetch`")

    names = [n.strip() for n in a.detectors.split(",") if n.strip()] or list(DETECTORS)
    docs = sample(a.csv, a.human_per_domain, a.ai_per_group, a.seed)

    # A small labelled probe for orientation, balanced so the direction test is
    # meaningful. Reused for every detector.
    rnd = random.Random(a.seed)
    h = [d for d in docs if d["model"] == "human"]
    m = [d for d in docs if d["model"] != "human"]
    probe = rnd.sample(h, 40) + rnd.sample(m, 40)
    probe_texts = [d["text"] for d in probe]
    probe_is_machine = [d["model"] != "human" for d in probe]

    results = {}
    for name in names:
        print(f"\n--- {name} ---", flush=True)
        t0 = time.time()
        try:
            det = build(name)
            # Contamination first: a detector fine-tuned on RAID is being
            # evaluated here on its own training data, and belongs in a
            # different table rather than at the top of this one.
            mid = getattr(det, "model_id", None) or getattr(
                getattr(det, "inner", None), "model_id", "")
            contam = check(mid) if mid else {
                "level": "unknown", "datasets": [], "note": "no model id resolved"}
            print(f"  [{BADGE[contam['level']]}] {contam['note']}", flush=True)
            sign = orient(det, probe_texts, probe_is_machine)
            if sign < 0:
                print("  label polarity reversed -> sign flipped", flush=True)
            det = OrientedDetector(det, sign)
            r = measure(det, docs, a.fpr, a.attacks, a.rate, a.seed)
            r["sign_flipped"] = sign < 0
            r["contamination"] = contam
            results[name] = r
            line = (f"  clean {r['clean']:.4f}  AUROC {r['auroc']:.4f}  "
                    f"distinct {r['distinct_frac']:.1%}  "
                    f"worst-domain FPR {r['worst_domain_fpr']:.1%} ({r['worst_domain']})")
            if a.attacks:
                line += (f"\n  attacks: mean {r['attack_mean']:.4f}  "
                         f"worst {r['attack_worst']:.4f} ({r['attack_worst_name']})")
            print(line + f"\n  [{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    ok = {k: v for k, v in results.items() if "error" not in v}

    def table(items, title):
        if not items:
            return
        print(f"\n{title}")
        print(f"{'detector':<26}{'clean':>8}{'AUROC':>8}{'distinct':>10}{'worstFPR':>10}"
              + (f"{'atk mean':>10}" if a.attacks else ""))
        for k, v in sorted(items, key=lambda kv: -kv[1]["clean"]):
            row = (f"{k:<26}{v['clean']:>8.4f}{v['auroc']:>8.4f}"
                   f"{v['distinct_frac']:>9.1%}{v['worst_domain_fpr']:>10.1%}")
            if a.attacks:
                row += f"{v['attack_mean']:>10.4f}"
            print(row)

    # Split by contamination, because ranking an in-domain model against
    # out-of-domain ones is not a comparison, it is a category error.
    def lvl(v):
        return (v.get("contamination") or {}).get("level", "unknown")

    table([kv for kv in ok.items() if lvl(kv[1]) in ("none", "unknown")],
          "RANKED - no RAID overlap declared")
    table([kv for kv in ok.items() if lvl(kv[1]) == "partial"],
          "SEPARATE - training data overlaps a RAID domain, read with care")
    table([kv for kv in ok.items() if lvl(kv[1]) == "direct"],
          "NOT COMPARABLE - trained ON RAID, so this is in-domain performance")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"fpr": a.fpr, "rate": a.rate, "seed": a.seed, "n_docs": len(docs),
               "attacks": a.attacks, "results": results}, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
