# Phase 1 Run Index

Phase 1 covers the first local study on `Qwen/Qwen3-0.6B-Base`.

## Run Table

| Run | Status | Method | Memory (bytes) | Perplexity | Quick read |
|-----|--------|--------|----------------|------------|------------|
| R1 | completed | full precision | 1192099840 | 16.8447 | baseline |
| R2 | completed | RTN 4-bit | 307304448 | 30.5169 | main quantized baseline |
| R3 | blocked | GPTQ 4-bit | n/a | n/a | deferred to another machine |
| R4 | completed | uniform SVD rank 4 | 307550208 | 30.5577 | repair too weak |
| R5 | completed | uniform SVD rank 8 | 307795968 | 30.4654 | slight recovery |
| R6 | completed | uniform SVD rank 16 | 308287488 | 30.3155 | moderate recovery |
| R7 | completed | uniform SVD rank 32 | 309270528 | 29.9848 | best repair result |
| R8 | completed | bits-only budget match to R4 | 307304448 | 30.5169 | budget too small to upgrade any layer |
| R9 | dry_run | bits-only budget match to R5 | n/a | n/a | not executed |
| R10 | completed | bits-only budget match to R6 | 307304448 | 30.5169 | still too small to upgrade any layer |
| R11 | completed | bits-only budget match to R7 | 308877312 | 28.8618 | first meaningful bits-only win |

## Main Takeaway

Uniform low-rank repair helps as rank increases, but once a meaningful bits-only comparator exists, extra bits beat uniform repair on this setup.

## Useful Files

- [phase1_summary.md](phase1_summary.md): generated summary table
- [../../docs/experiments/phase1_results.md](../../docs/experiments/phase1_results.md): curated interpretation
- [../../docs/experiments/experiment_journal.md](../../docs/experiments/experiment_journal.md): detailed run-by-run notes
