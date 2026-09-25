"""Adversarial perturbations.

RAID does not ship its attack implementations, so these are independent
versions of the same ideas. They measure the cliff; they are not byte-identical
to RAID's own, and results here are not directly comparable to leaderboard
figures.

Each attack takes (text, rate, rng) and returns perturbed text. `rate` is the
per-character or per-token probability of perturbation; 0.15 is the default
used for the published numbers.

Two attacks here need more than string manipulation and are optional as a
result: `synonym` needs WordNet (`pip install nltk` plus
`nltk.download("wordnet")`), and `paraphrase` needs a seq2seq model (see
`benchmark/paraphrase.py`). Both are skipped with a warning if their dependency
is missing, so the rest of the suite still runs.
"""
import argparse
import csv
import random

csv.field_size_limit(10_000_000)

# Latin -> visually identical Cyrillic/Greek. Mirrors the fold table in
# app/services/ai_detection_service.py so the test exercises the real defence.
HOMOGLYPH = {
    "a": "\u0430", "c": "\u0441", "e": "\u0435", "o": "\u043e", "p": "\u0440",
    "x": "\u0445", "y": "\u0443", "i": "\u0456", "j": "\u0458", "s": "\u0455",
    "A": "\u0410", "B": "\u0412", "C": "\u0421", "E": "\u0415", "H": "\u041d",
    "K": "\u041a", "M": "\u041c", "O": "\u041e", "P": "\u0420", "T": "\u0422",
    "X": "\u0425", "Y": "\u0423",
}
ZERO_WIDTH = "\u200b"


def homoglyph(text, rate, rnd):
    return "".join(
        HOMOGLYPH[ch] if ch in HOMOGLYPH and rnd.random() < rate else ch
        for ch in text
    )


def zero_width(text, rate, rnd):
    out = []
    for ch in text:
        out.append(ch)
        if rnd.random() < rate:
            out.append(ZERO_WIDTH)
    return "".join(out)


def whitespace(text, rate, rnd):
    """Space before punctuation - the bypass closed by commit ea48a2a."""
    out = []
    for ch in text:
        if ch in ".,;:!?" and rnd.random() < rate:
            out.append(" ")
        out.append(ch)
    return "".join(out)


_STOP = {
    "the","a","an","and","or","but","if","of","to","in","on","at","for","with",
    "as","by","is","are","was","were","be","been","being","it","its","this",
    "that","these","those","from","has","have","had","not","no","do","does",
    "did","will","would","can","could","should","may","might","must","than",
    "then","so","such","there","their","they","he","she","we","you","i",
}


def _wordnet():
    from nltk.corpus import wordnet as wn
    wn.synsets("test")          # forces the corpus load, raises if absent
    return wn


def synonym(text, rate, rnd):
    """Replace words with WordNet synonyms.

    Unlike every other attack here this changes the words themselves, not their
    encoding, so input normalisation cannot undo it. Function words are skipped -
    substituting those produces obvious nonsense rather than a plausible edit,
    which would overstate the attack.
    """
    wn = _wordnet()
    out = []
    for tok in text.split(" "):
        core = tok.strip(".,;:!?()[]\"'").lower()
        if (len(core) < 4 or core in _STOP or not core.isalpha()
                or rnd.random() >= rate):
            out.append(tok)
            continue
        lemmas = []
        for syn in wn.synsets(core)[:4]:
            for lem in syn.lemmas():
                w = lem.name().replace("_", " ")
                if w.lower() != core and " " not in w:
                    lemmas.append(w)
        if not lemmas:
            out.append(tok)
            continue
        rep = rnd.choice(lemmas)
        if core[0].isupper() or tok[:1].isupper():
            rep = rep[:1].upper() + rep[1:]
        out.append(tok.replace(tok.strip(".,;:!?()[]\"'"), rep, 1))
    return " ".join(out)


def paraphrase(text, rate, rnd):
    """Rewrite a fraction of sentences with a seq2seq paraphraser.

    Delegates to benchmark.paraphrase, which holds the model. Like `synonym`,
    this alters meaning-bearing content rather than encoding, so normalisation
    offers no defence.
    """
    from .paraphrase import paraphrase_text
    return paraphrase_text(text, rate, rnd)


def upper_lower(text, rate, rnd):
    """Randomly flip letter case."""
    return "".join(
        (ch.lower() if ch.isupper() else ch.upper()) if ch.isalpha() and rnd.random() < rate
        else ch
        for ch in text
    )


_ARTICLES = {"the", "a", "an"}


def article_deletion(text, rate, rnd):
    out = []
    for tok in text.split(" "):
        if tok.strip(".,;:!?").lower() in _ARTICLES and rnd.random() < rate:
            continue
        out.append(tok)
    return " ".join(out)


def number_swap(text, rate, rnd):
    return "".join(
        (str(rnd.randint(0, 9)) if rnd.random() < rate else ch) if ch.isdigit() else ch
        for ch in text
    )


def misspelling(text, rate, rnd):
    """Transpose two adjacent letters inside a word."""
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 3 and w.isalpha() and rnd.random() < rate:
            j = rnd.randrange(len(w) - 1)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


# US -> UK, the usual suspects. RAID's alternative_spelling attack is the same idea.
_US_UK = {
    "color": "colour", "colors": "colours", "favor": "favour", "flavor": "flavour",
    "honor": "honour", "labor": "labour", "neighbor": "neighbour", "behavior": "behaviour",
    "center": "centre", "centers": "centres", "theater": "theatre", "meter": "metre",
    "organize": "organise", "organized": "organised", "recognize": "recognise",
    "realize": "realise", "realized": "realised", "analyze": "analyse",
    "defense": "defence", "offense": "offence", "license": "licence",
    "traveling": "travelling", "traveled": "travelled", "canceled": "cancelled",
    "modeling": "modelling", "program": "programme", "gray": "grey",
    "aluminum": "aluminium", "catalog": "catalogue", "dialog": "dialogue",
}


def alternative_spelling(text, rate, rnd):
    words = text.split(" ")
    for i, w in enumerate(words):
        core = w.strip(".,;:!?()").lower()
        if core in _US_UK and rnd.random() < rate:
            words[i] = w.replace(core, _US_UK[core]) if core in w else w
            low = w.lower()
            if core in low:
                idx = low.index(core)
                words[i] = w[:idx] + _US_UK[core] + w[idx + len(core):]
    return " ".join(words)


def insert_paragraphs(text, rate, rnd):
    out = []
    for part in text.split(". "):
        out.append(part)
        out.append(".\n\n" if rnd.random() < rate else ". ")
    return "".join(out[:-1]) if out else text


ATTACKS = {
    "homoglyph": homoglyph,
    "synonym": synonym,
    "paraphrase": paraphrase,
    "zero_width": zero_width,
    "whitespace": whitespace,
    "upper_lower": upper_lower,
    "article_deletion": article_deletion,
    "number_swap": number_swap,
    "misspelling": misspelling,
    "alternative_spelling": alternative_spelling,
    "insert_paragraphs": insert_paragraphs,
}
# `synonym` and `paraphrase` are the only two that change words rather than
# encoding. They are also the two that input normalisation cannot defend against,
# which is why they are worth the extra dependency.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--attack", choices=sorted(ATTACKS), required=True)
    p.add_argument("--rate", type=float, default=0.15)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rnd = random.Random(a.seed)
    fn = ATTACKS[a.attack]
    rows, cols = [], None
    with open(a.csv, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        for i, r in enumerate(rd):
            if a.limit and i >= a.limit:
                break
            r["generation"] = fn(r.get("generation") or "", a.rate, rnd)
            r["attack"] = a.attack
            r["id"] = f"{r['id']}::{a.attack}"  # keep ids distinct from the clean run
            rows.append(r)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {a.out}  (attack={a.attack} rate={a.rate})")


if __name__ == "__main__":
    main()
