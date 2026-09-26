# Preprint source

`main.tex` — standalone LaTeX, no custom class or style files. Compiles with
`pdflatex main.tex` (twice, for references) or by uploading to Overleaf.

## Before submitting

- **Confirm the author line.** It currently reads `Dipak Bhosale, Lacewing
  Technologies LLC` taken from the repository's git identity and licence. Check
  the name, affiliation and contact address, and add co-authors.
- **Check the conflict-of-interest paragraph** in the introduction reads the way
  you want it to. One of the nine detectors evaluated is published by the same
  organisation, the paper says so in Section 1, and Section 7 reports that our
  detector has the worst worst-domain false-positive rate of the nine. That
  combination is what makes the rest credible; softening it would cost more than
  it gains.
- **Categories:** `cs.CL` primary, `cs.LG` cross-list.
- **Verify every number against the harness** before it goes out. Each table is
  reproducible from the repository at seed 0; regenerate rather than trust the
  transcription.

## What the paper does and does not claim

It is a measurement paper about evaluation artifacts, not a claim that any
detector is good. Every headline is of the form "this factor moved the metric by
more than the differences used to rank detectors" — reporting format, input
normalisation, provenance, and attack strength. The limitations section is long
on purpose and should not be trimmed: the largest single weakness is that four of
the six findings use one detector.
