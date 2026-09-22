# Why robustness results are not tabulated here

This harness measures how detector accuracy degrades under mechanical
perturbation of the input. Running it prints those numbers. This repository does
not publish them as a table, and the reasoning is worth stating plainly rather
than leaving as an omission.

The attack *categories* are not novel or secret — they come from RAID's own
published taxonomy, and anyone working in this area knows them. What a results
table adds is the combination of a specific perturbation, a specific strength,
and a measured evasion rate against a named, currently-deployed detector. That
combination is materially more useful to someone trying to evade detection than
it is to someone trying to build a better one, and the asymmetry gets worse when
a mitigation exists but is only partial.

So the position taken here is:

- **The harness is public.** Anyone evaluating a detector — their own or
  someone else's — can measure robustness and see exactly how it was measured.
  That is the part with research value.
- **Per-attack evasion rates against deployed systems are not.** Those belong
  in a disclosure to the affected vendor, or in a write-up published once
  mitigations are in place and measured.

## If you are evaluating a detector

Run it. The numbers print. Read them against these two limits:

**Synonym substitution and paraphrase are not implemented.** Both need a model
or a thesaurus, and both are more damaging than any perturbation included here.
Any robustness figure this harness produces is therefore an **upper bound**. A
detector that survives these attacks is unproven, not robust.

**Mitigations are usually partial.** A defence that recovers most of the loss
still leaves a gap, and a gap that a table makes precise is a gap that is easier
to aim at. Report what you measured, including what you did not fix.

## Reporting an issue

If you find a weakness in a detector measured by this harness, please contact
the vendor before publishing specifics.
