"""Detectors under test.

A detector is anything with a `name` and a `score(texts) -> list[float]`, where
a HIGHER score means MORE likely machine-generated. The scale does not matter:
every metric in this benchmark is rank-based or fits its own threshold, so a
detector may return log-odds, a probability, or an arbitrary monotonic score.

Add your own by subclassing Detector and registering it in DETECTORS.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .normalize import normalize


class Detector:
    name = "detector"

    def score(self, texts: list[str]) -> list[float]:
        raise NotImplementedError


class HFSequenceClassifier(Detector):
    """Any HuggingFace sequence-classification detector.

    Scores with the LOG-ODDS MARGIN (logit_ai - logit_human), not the softmax
    probability. This matters more than it sounds.

    A confidently fine-tuned classifier saturates: softmax returns 0.99999999,
    and once that is rounded for display it becomes exactly 1.0. Measured on
    2,520 RAID documents with the model below, 77.1% of documents collapsed onto
    a handful of values. Any metric that fits a threshold then has to split a
    mass of exact ties, and accuracy@5%FPR fell from 0.7144 to 0.5273 purely
    from that rounding - in two domains it fell to zero. The margin is unbounded
    and does not saturate, so it preserves the ranking the model actually
    produced. Report it, or you will measure your rounding instead of your model.
    """

    def __init__(self, model_id: str, name: str | None = None,
                 ai_label_hint: str = "AI", max_length: int = 512,
                 batch_size: int = 32, device: str | None = None):
        self.model_id = model_id
        self.name = name or model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = None
        self._model = None
        self._ai_idx = 1
        self._ai_label_hint = ai_label_hint

    def _load(self):
        if self._model is not None:
            return
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id).to(self.device)
        self._model.eval()
        # Resolve the AI class from the checkpoint's own label map rather than
        # assuming index 1 - not every detector orders them the same way.
        id2label = getattr(self._model.config, "id2label", {}) or {}
        for idx, label in id2label.items():
            s = str(label).upper()
            if s == "LABEL_1" or self._ai_label_hint.upper() in s or "FAKE" in s \
                    or "MACHINE" in s or "GENERATED" in s or "CHATGPT" in s:
                self._ai_idx = int(idx)
                break

    def score(self, texts: list[str]) -> list[float]:
        self._load()
        human_idx = 1 - self._ai_idx
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            enc = self._tok(chunk, return_tensors="pt", truncation=True,
                            max_length=self.max_length, padding=True).to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits.float()
            out.extend((logits[:, self._ai_idx] - logits[:, human_idx]).tolist())
        return out


class NormalizingDetector(Detector):
    """Wraps any detector with scoring-path input normalisation.

    This is the defence, not an attack: NFKC, confusable folding, invisible
    character removal, whitespace repair and case repair, applied before the
    model sees the text. Measure a detector both ways to see how much of its
    apparent fragility is the model and how much is unnormalised input.
    """

    def __init__(self, inner: Detector, case_repair: bool = True,
                 name: str | None = None):
        self.inner = inner
        self.case_repair = case_repair
        self.name = name or f"{inner.name}+normalised"

    def score(self, texts: list[str]) -> list[float]:
        return self.inner.score([normalize(t, self.case_repair) for t in texts])


# The detector this benchmark was built to measure. The weights are public and
# ungated; sha256 of model.safetensors is
# 9177c456a91e41afc6e03583d81a63a5b81d5d0a96ecfd783cd5b8c234418443
# (1,740,304,440 bytes), which is the exact checkpoint the published numbers
# were produced from.
TEXTSIGHT_V23 = "textsightai/textsight-detector-v23-custom"

DETECTORS = {
    # The bare checkpoint, no input handling. Useful as a baseline, but note
    # that no sane production system scores raw user input this way.
    "textsight-v23": lambda: HFSequenceClassifier(TEXTSIGHT_V23, "textsight-v23"),
    # The same weights behind scoring-path normalisation - the configuration a
    # deployed detector should actually be judged in.
    "textsight-v23-normalised": lambda: NormalizingDetector(
        HFSequenceClassifier(TEXTSIGHT_V23, "textsight-v23"),
        name="textsight-v23-normalised"),
    # Open baselines, all standard sequence classifiers on the Hub. Download
    # counts are a rough proxy for how widely each is actually used.
    # RADAR also appears on the RAID leaderboard, which gives an external
    # reference point for this harness.
    "radar": lambda: HFSequenceClassifier(
        "TrustSafeAI/RADAR-Vicuna-7B", "radar"),
    "openai-roberta-base": lambda: HFSequenceClassifier(
        "openai-community/roberta-base-openai-detector", "openai-roberta-base"),
    "openai-roberta-large": lambda: HFSequenceClassifier(
        "openai-community/roberta-large-openai-detector", "openai-roberta-large"),
    "hc3-roberta": lambda: HFSequenceClassifier(
        "Hello-SimpleAI/chatgpt-detector-roberta", "hc3-roberta"),
    "piratexx": lambda: HFSequenceClassifier(
        "PirateXX/AI-Content-Detector", "piratexx"),
    "roberta-mixed": lambda: HFSequenceClassifier(
        "andreas122001/roberta-mixed-detector", "roberta-mixed"),
    "e5-small-lora": lambda: HFSequenceClassifier(
        "MayZhou/e5-small-lora-ai-generated-detector", "e5-small-lora"),
    # NOT included: desklib/ai-text-detector-v1.01 ships a custom
    # DesklibAIDetectionModel class and needs trust_remote_code=True, so it does
    # not load through AutoModelForSequenceClassification. Excluded rather than
    # given a bespoke adapter that could quietly differ from how it is meant to
    # be run.
}


def orient(det: Detector, texts: list[str], is_machine: list[bool]) -> int:
    """Return +1 or -1 so that higher scores mean more machine-generated.

    Every detector on the Hub orders its labels differently, and several use bare
    LABEL_0/LABEL_1 with no documented polarity. Guessing from the label map is
    unreliable, so the direction is established empirically on a small labelled
    probe: if the scores rank human above machine, the sign is flipped. Without
    this a comparison can silently invert a detector and report it as far worse
    than it is, which would make the whole table meaningless.
    """
    s = det.score(texts)
    pos = [v for v, m in zip(s, is_machine) if m]
    neg = [v for v, m in zip(s, is_machine) if not m]
    if not pos or not neg:
        return 1
    # Mann-Whitney direction: is a machine document usually scored higher?
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    auc = (wins + 0.5 * ties) / (len(pos) * len(neg))
    return 1 if auc >= 0.5 else -1


class OrientedDetector(Detector):
    """A detector with its sign fixed so higher always means machine-generated."""

    def __init__(self, inner: Detector, sign: int):
        self.inner = inner
        self.sign = sign
        self.name = inner.name

    def score(self, texts: list[str]) -> list[float]:
        return [self.sign * v for v in self.inner.score(texts)]


def build(name: str) -> Detector:
    if name not in DETECTORS:
        raise SystemExit(f"unknown detector {name!r}; have: {', '.join(DETECTORS)}")
    return DETECTORS[name]()
