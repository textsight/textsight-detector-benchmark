"""Sentence-level paraphrase attack.

Paraphrase is the attack that matters most and the one every other perturbation
here is a cheap imitation of. It changes the words rather than their encoding, so
input normalisation offers no defence at all, and it is what anyone seriously
trying to pass machine-generated text off as human would actually reach for.

Implementation is deliberately plain: split into sentences, rewrite a fraction of
them with a public seq2seq paraphraser, rejoin. RAID's own paraphrase attack uses
DIPPER, a paragraph-level model, so numbers here are not comparable to theirs -
this measures the same idea, not the same tool.

The model is loaded once, lazily, and reused. Sentences are batched, so the cost
is a few minutes over a few thousand documents rather than hours.
"""
from __future__ import annotations

import re

_MODEL_ID = "humarin/chatgpt_paraphraser_on_T5_base"
_SENT = re.compile(r"(?<=[.!?])\s+")

_state: dict = {"tok": None, "model": None, "device": None, "failed": False}


def _load():
    if _state["model"] is not None or _state["failed"]:
        return _state["model"] is not None
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _state["tok"] = AutoTokenizer.from_pretrained(_MODEL_ID)
        _state["model"] = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_ID).to(dev).eval()
        _state["device"] = dev
        return True
    except Exception as e:                                    # noqa: BLE001
        print(f"  [paraphrase unavailable: {type(e).__name__}: {str(e)[:90]}]")
        _state["failed"] = True
        return False


def _rewrite(sentences: list[str], batch_size: int = 32) -> list[str]:
    import torch
    tok, model, dev = _state["tok"], _state["model"], _state["device"]
    out: list[str] = []
    for i in range(0, len(sentences), batch_size):
        chunk = [f"paraphrase: {s}" for s in sentences[i:i + batch_size]]
        enc = tok(chunk, return_tensors="pt", truncation=True,
                  max_length=96, padding=True).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=96, num_beams=1,
                                 do_sample=True, top_p=0.95, temperature=1.0)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def paraphrase_text(text: str, rate: float, rnd) -> str:
    """Rewrite roughly `rate` of the sentences in `text`."""
    if not _load():
        return text
    sents = [s for s in _SENT.split(text) if s.strip()]
    if not sents:
        return text
    idx = [i for i in range(len(sents)) if rnd.random() < rate]
    if not idx:
        return text
    rewritten = _rewrite([sents[i] for i in idx])
    for i, new in zip(idx, rewritten):
        if new.strip():
            sents[i] = new.strip()
    return " ".join(sents)


def paraphrase_corpus(texts: list[str], rate: float, rnd,
                      batch_size: int = 48) -> list[str]:
    """Paraphrase many documents, batching sentences across all of them.

    Per-document batching wastes most of a GPU batch on short documents. Pooling
    every selected sentence from the whole corpus into one queue is several times
    faster and produces identical output.
    """
    if not _load():
        return list(texts)
    split = [[s for s in _SENT.split(t) if s.strip()] for t in texts]
    picks, queue = [], []
    for d, sents in enumerate(split):
        for i in range(len(sents)):
            if rnd.random() < rate:
                picks.append((d, i))
                queue.append(sents[i])
    if not queue:
        return list(texts)
    done = _rewrite(queue, batch_size=batch_size)
    for (d, i), new in zip(picks, done):
        if new.strip():
            split[d][i] = new.strip()
    return [" ".join(s) for s in split]
