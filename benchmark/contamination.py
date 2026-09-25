"""Training-data contamination checks.

This benchmark measures on RAID's labelled `train_none` split, because RAID's
test splits are blind. That makes one mistake very easy and very damaging: a
detector that was FINE-TUNED on RAID is then being evaluated on its own training
data, and will post a score that has nothing to do with generalisation.

It is not a hypothetical. `MayZhou/e5-small-lora-ai-generated-detector` declares
`datasets: [liamdugan/raid]` and scored 0.9364 here - top of the table, ahead of
every out-of-domain model, which is exactly what in-domain evaluation looks like.

So contamination is checked automatically from each model's declared metadata
rather than left to whoever reads the table. Declared metadata is not proof: a
model can omit its datasets, or describe them only in prose. Treat a clean result
as "nothing declared", not as "not contaminated".
"""
from __future__ import annotations

import json
import urllib.request

# Datasets that are RAID, or are RAID's own sources closely enough that training
# on them gives an in-domain advantage on specific RAID domains.
_DIRECT = ("liamdugan/raid", "raid-bench", "raid_bench")
_PARTIAL = {
    "gfissore/arxiv-abstracts": "overlaps RAID's `abstracts` domain",
    "ml-arxiv-papers": "overlaps RAID's `abstracts` domain",
    "wikipedia": "overlaps RAID's `wiki` domain",
    "cnn_dailymail": "overlaps RAID's `news` domain",
    "recipe": "overlaps RAID's `recipes` domain",
    "imdb": "overlaps RAID's `reviews` domain",
}


def declared_datasets(model_id: str, timeout: int = 20) -> list[str]:
    try:
        with urllib.request.urlopen(
                f"https://huggingface.co/api/models/{model_id}", timeout=timeout) as r:
            d = json.load(r)
    except Exception:
        return []
    ds = (d.get("cardData") or {}).get("datasets") or []
    if isinstance(ds, str):
        ds = [ds]
    return [str(x) for x in ds]


def check(model_id: str) -> dict:
    """Returns {'level': 'direct'|'partial'|'none'|'unknown', 'datasets', 'note'}."""
    ds = declared_datasets(model_id)
    if not ds:
        return {"level": "unknown", "datasets": [], "note":
                "no datasets declared - contamination cannot be ruled out"}
    low = [x.lower() for x in ds]
    for d in low:
        if any(k in d for k in _DIRECT):
            return {"level": "direct", "datasets": ds, "note":
                    "TRAINED ON RAID - evaluated here on its own training data, "
                    "not comparable with out-of-domain models"}
    notes = [f"{d} {why}" for d in low for k, why in _PARTIAL.items() if k in d]
    if notes:
        return {"level": "partial", "datasets": ds, "note": "; ".join(sorted(set(notes)))}
    return {"level": "none", "datasets": ds, "note": "no overlap declared"}


BADGE = {"direct": "IN-DOMAIN", "partial": "partial-overlap",
         "none": "out-of-domain", "unknown": "undeclared"}
