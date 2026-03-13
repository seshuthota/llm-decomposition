# Phase 1 Results

This document consolidates the Phase 1 summary and the decision memo into one readable status document.

## Scope

Phase 1 was run on:

- model: `Qwen/Qwen3-0.6B-Base`
- local quantizer: `RTN`
- evaluation focus: perplexity, memory, layerwise damage, and early equal-budget comparisons

The original goal was to test whether low-rank repair looked promising enough to justify a larger hybrid-compression effort.

## Run Summary

| Run | Method | Memory (bytes) | Perplexity | Notes |
|-----|--------|----------------|------------|-------|
| R1 | full precision | 1192099840 | 16.8447 | baseline |
| R2 | RTN 4-bit | 307304448 | 30.5169 | quantized baseline |
| R3 | GPTQ 4-bit | n/a | n/a | deferred to another machine |
| R4 | uniform SVD rank 4 | 307550208 | 30.5577 | slightly worse than `R2` |
| R5 | uniform SVD rank 8 | 307795968 | 30.4654 | small recovery |
| R6 | uniform SVD rank 16 | 308287488 | 30.3155 | clearer recovery |
| R7 | uniform SVD rank 32 | 309270528 | 29.9848 | best repair result in Phase 1 |
| R8 | bits-only budget match to `R4` | 307304448 | 30.5169 | budget too small to upgrade any layer |
| R9 | bits-only budget match to `R5` | n/a | n/a | not run |
| R10 | bits-only budget match to `R6` | 307304448 | 30.5169 | budget still too small to upgrade any layer |
| R11 | bits-only budget match to `R7` | 308877312 | 28.8618 | first meaningful bits-only comparison |

Generated summary:

- [../../results/phase1/phase1_summary.md](../../results/phase1/phase1_summary.md)

## What We Learned

### 1. Quantization Damage Is Real and Concentrated

`R2` showed a large memory reduction but a sharp quality drop:

- `R1` full precision: `16.8447` perplexity
- `R2` RTN 4-bit: `30.5169` perplexity

Layerwise profiling showed that the damage was not uniform. Later `mlp.down_proj` layers were especially sensitive, with some `self_attn.o_proj` layers also standing out.

### 2. RTN Residuals Are Not Strongly Low-Rank at Tiny Ranks

The top damaged RTN residuals did not show especially fast singular value decay at very small ranks. That weakened the original "tiny repair is enough" story.

### 3. Uniform Low-Rank Repair Is Not Dead, but It Is Weak Early

The repair trend was monotonic after rank 4:

- `R4` rank 4: worse than baseline
- `R5` rank 8: slight improvement
- `R6` rank 16: moderate improvement
- `R7` rank 32: best repair result

So low-rank repair can recover some quality, but useful gains require non-trivial rank.

### 4. Once Bits-Only Became Real, It Won

`R8` and `R10` were not meaningful because the bits-only comparator was too coarse to spend such small budgets.

`R11` was the decisive comparison:

- it upgraded `model.layers.27.mlp.down_proj`
- it used a comparable budget range to `R7`
- it achieved `28.8618` perplexity versus `29.9848` for `R7`

That is the clearest signal collected so far.

## GPTQ Status

`R3` could not be completed locally.

Reason:

- the attempted GPTQ dependency path required CUDA kernels targeting `sm_80+`
- the local GPU is `NVIDIA GeForce GTX 1660 SUPER` (`sm_75`)

Conclusion:

- GPTQ should be run on another machine
- RTN is still sufficient for local allocator development

## Decision

The project mainline should pivot from **uniform low-rank repair** to **bit allocation / mixed-precision quantization**.

Low-rank repair remains:

- a useful baseline,
- a diagnostic tool,
- and a possible second-stage hybrid extension

but it is no longer the lead method direction on current evidence.

## Immediate Next Step

Build a finer-grained bits-only allocator so equal-budget comparisons are meaningful even at smaller memory increments.

That should be the next implementation milestone before any larger hybrid push.
