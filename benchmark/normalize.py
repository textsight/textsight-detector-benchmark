"""Input normalisation for the scoring path.

Most of the cheap attacks against a transformer detector work the same way:
leave the text looking identical to a human reader while changing the byte
sequence, so the subword tokenizer produces fragments the model never saw in
training. A Cyrillic "a", a zero-width space between letters, a flipped capital
in the middle of a word - none of them change what the text says, and all of
them change what the model reads.

Normalising the input before scoring closes that class of attack directly, and
costs almost nothing on clean text. This module is the defence measured in the
README; wrap any detector with NormalizingDetector to apply it.

Order matters. NFKC first (compatibility forms), then confusable folding, then
invisible-character removal, then whitespace, then case repair.
"""
from __future__ import annotations

import re
import unicodedata

# Invisible and format characters: zero-width space/non-joiner/joiner, LTR/RTL
# marks, word joiner, BOM, soft hyphen.
_ZERO_WIDTH = re.compile("[​-‏⁠﻿­]")
# Non-ASCII spaces, including NBSP, en/em spaces, narrow NBSP, ideographic.
_UNICODE_SPACE = re.compile("[   -   　]")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,;:!?%）\)\]\}])")
_SPACE_AFTER_OPEN = re.compile(r"([（\(\[\{])[ \t]+")
_SPACE_BEFORE_CLITIC = re.compile(r"[ \t]+('(?:s|d|t|m|re|ve|ll)\b|n't\b)",
                                  re.IGNORECASE)
_RUNS = re.compile(r"[ \t]{2,}")

# Latin lookalikes drawn from Cyrillic and Greek. Folding these is what defeats
# homoglyph substitution.
_CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w", "ԁ": "d",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "У": "Y", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    "α": "a", "ο": "o", "ρ": "p", "υ": "u", "χ": "x",
    "ι": "i", "κ": "k", "ν": "v",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
_CONFUSABLE_RE = re.compile("[" + "".join(_CONFUSABLES) + "]")
_WORD = re.compile(r"\w+", re.UNICODE)
_LATIN = re.compile(r"[A-Za-z]")


def fold_confusables(text: str) -> str:
    """Fold Cyrillic/Greek lookalikes to Latin inside MIXED-SCRIPT WORDS only.

    The discriminator is mixed script within a word, not the document's overall
    share of Cyrillic. A document-level ratio test fails as soon as an attacker
    substitutes every instance of a target letter, which pushes the ratio past
    any threshold you pick. Genuine Russian or Greek keeps whole words in one
    script, so they are left exactly as they are and real non-Latin documents
    are unaffected.
    """
    if not text or not _CONFUSABLE_RE.search(text):
        return text

    def fix(m):
        w = m.group()
        if not _CONFUSABLE_RE.search(w) or not _LATIN.search(w):
            return w
        return _CONFUSABLE_RE.sub(lambda c: _CONFUSABLES[c.group()], w)

    return _WORD.sub(fix, text)


_WORD_LETTERS = re.compile(r"[A-Za-z]+")


def repair_case(text: str) -> str:
    """Lowercase the tail of words carrying an anomalous internal capital.

    ALL-CAPS acronyms and ordinary Capitalised words are left alone. This cannot
    fully undo a case-flipping attack - a flipped FIRST letter is
    indistinguishable from legitimate capitalisation - but it removes the noise
    the tokenizer chokes on. It also flattens intercaps brand names
    (iPhone -> iphone), which is a deliberate trade on the scoring path.
    """
    if not text:
        return text

    def fix(m):
        w = m.group()
        if len(w) < 2 or w.isupper():
            return w
        if any(c.isupper() for c in w[1:]):
            return w[0] + w[1:].lower()
        return w

    return _WORD_LETTERS.sub(fix, text)


def normalize(text: str, case_repair: bool = True) -> str:
    """Full scoring-path normalisation. Idempotent."""
    if not text:
        return text
    t = unicodedata.normalize("NFKC", text)
    t = fold_confusables(t)
    if case_repair:
        t = repair_case(t)
    t = _ZERO_WIDTH.sub("", t)
    t = _UNICODE_SPACE.sub(" ", t)
    t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
    t = _SPACE_AFTER_OPEN.sub(r"\1", t)
    t = _SPACE_BEFORE_CLITIC.sub(r"\1", t)
    t = _RUNS.sub(" ", t)
    return t.strip()
