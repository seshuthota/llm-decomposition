# Proposal History

This project has already gone through one important framing change.

## Initial Framing

The initial version of the project focused on **budget-aware low-rank residual allocation** for quantized language models.

Document:

- [initial_proposal.md](initial_proposal.md)

Core idea:

- start from a quantized baseline
- add low-rank residual repair to damaged layers
- study whether rank is a better use of memory than more bits

## What Changed

Phase 1 experiments showed that the broad hybrid direction was still interesting, but the naive version of it was too optimistic:

- RTN quantization damage is clearly non-uniform across layers
- very small uniform low-rank repair did little
- higher-rank repair helped somewhat
- once a meaningful bits-only baseline existed, extra bits beat uniform repair clearly

That shifted the project from a "repair-first" story to an **allocation-first** story.

## Current Framing

The active proposal now treats **bit allocation / mixed-precision quantization** as the mainline method and keeps low-rank repair as:

- a secondary comparison,
- a diagnostic tool,
- and a possible second-stage hybrid extension after the best bit allocations are made.

Document:

- [current_proposal.md](current_proposal.md)

## Working Narrative

The core research question is now:

> Under a fixed memory budget, when should the next byte go to higher precision, and when should it go to low-rank repair?

That framing is more specific, more defensible, and better aligned with the evidence collected so far.
