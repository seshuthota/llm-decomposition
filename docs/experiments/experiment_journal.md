# Experiment Journal

This file records each experiment run, what was executed, what was observed, and what changed as a result. The goal is to keep a durable, human-readable history alongside the raw JSON outputs in `results/`.

## Phase 1

### Run R1

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `full_precision`
- Script: `./scripts/run_r1_full_precision.sh`
- Log: `results/logs/phase1_20260311_134453.log`
- Metrics: `results/phase1/R1/metrics.json`

Command:

```bash
./scripts/run_r1_full_precision.sh
```

Result:

- status: `completed`
- device: `cuda`
- dtype: `bfloat16`
- perplexity: `16.844704810407233`
- evaluated tokens: `297913`
- memory total bytes: `1192099840`
- latency ms/token: `0.6852036503778915`

Notes:

- The first run surfaced a data pipeline issue.
- The original sequence builder tokenized the entire evaluation split as one very large string before chunking.
- This triggered the tokenizer warning:
  - `Token indices sequence length is longer than the specified maximum sequence length for this model`
- The run still completed successfully and produced valid metrics, but the pipeline behavior was not clean.

Action taken:

- Updated `llm_decomposition/hf_utils.py` so text is tokenized sample-by-sample and concatenated at the token level before fixed-length chunking.
- This avoids constructing a single oversized token sequence during tokenization.

Interpretation:

- `R1` establishes the baseline quality and memory point for all later comparisons.
- This is the reference point for quantization loss and quality recovery.

### Run R2

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `RTN 4-bit`
- Script: `./scripts/run_r2_rtn_4bit.sh`
- Log: `results/logs/phase1_20260311_135448.log`
- Metrics: `results/phase1/R2/metrics.json`
- Layer errors: `results/phase1/R2/layer_errors.json`
- Residual profiles: `results/phase1/R2/residual_profiles.json`

Command:

```bash
./scripts/run_r2_rtn_4bit.sh
```

Result:

- status: `completed`
- device: `cuda`
- dtype: `bfloat16`
- perplexity: `30.516916135923783`
- evaluated tokens: `300979`
- memory total bytes: `307304448`
- memory metadata bytes: `9312256`
- latency ms/token: `0.6853402444656435`

Comparison vs R1:

- full-precision perplexity: `16.8447`
- RTN-4bit perplexity: `30.5169`
- full-precision memory: `1192099840` bytes
- RTN-4bit memory: `307304448` bytes

Observed effect:

- memory dropped substantially, from about `1.19 GB` to about `307 MB`
- quality dropped sharply, which means there is clear room for recovery methods to matter

Layerwise observations:

- the highest activation-space error is concentrated in later `mlp.down_proj` layers
- several `self_attn.o_proj` layers also appear near the top
- this supports the project hypothesis that quantization damage is non-uniform across layers

Residual profile observations:

- the top damaged layers do not look strongly low-rank under the current RTN setup
- example: `model.layers.27.mlp.down_proj`
  - rank-4 explained energy: `0.0306`
  - rank-8 explained energy: `0.0441`
  - rank-16 explained energy: `0.0670`
  - rank-32 explained energy: `0.1068`
- similar patterns appear across the other top-damage layers

Interpretation:

- this is an important result
- the residuals are concentrated in a subset of layers, which is good for targeted repair
- but the residual energy captured by very low ranks is currently modest, which weakens the simplest low-rank correction story
- this does not kill the project, but it raises the bar:
  - either the useful correction rank may need to be higher,
  - or mixed bit allocation may be more competitive than pure low-rank repair,
  - or GPTQ-style residuals may be more structured than RTN residuals

Actionable next questions:

1. Does `GPTQ 4-bit` produce residuals that are more compressible than `RTN 4-bit`?
2. Are the most damaged layers still the same under a stronger quantizer?
3. Is a small uniform low-rank repair already too weak at realistic ranks?

## Working Conclusions After R1-R2

- The harness is working for `full_precision` and `RTN`.
- The first baseline comparison is now established.
- The project already has one concrete empirical result:
  - quantization damage is non-uniform,
  - later MLP down projections are especially sensitive,
  - but RTN residuals do not appear strongly low-rank at very small ranks.
- GPTQ work should be deferred to a different machine with a GPU/runtime path compatible with the current GPTQ dependency stack.

## GPTQ Bring-up

### Run R3_Q17B (First GPTQ Baseline Attempt)

- Date: 2026-03-13
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `GPTQ 4-bit`
- Script: `./scripts/run_qwen3_1p7b_r3_gptq_modal.sh`
- Metrics: `results/modal/qwen3_1p7b_gptq_baselines/R3_Q17B/metrics.json`

Observed result:

- status: `completed`
- device: `cuda`
- dtype: `bfloat16`
- perplexity: `16103625.670221116`
- memory total bytes: `1357024282`

Observed problems:

- perplexity is catastrophically bad, so this baseline is not scientifically usable
- `layer_errors.json` only included `lm_head`, which means the baseline could not support a proper top-k GPTQ candidate pool

Root causes identified:

- GPTQ baseline execution was mutating the only loaded model and then trying to estimate layer stats / activation errors from that already-quantized model
- GPTQ runs were inheriting the general CUDA dtype preference path, which selected `bfloat16` rather than forcing `float16`

Action taken:

- patched the GPTQ runtime path to force `float16` on CUDA
- changed the GPTQ baseline to keep a separate untouched full-precision reference model
- generalized profiling so the quantized side is matched by module name rather than only `nn.Linear`
- disabled activation-heavy profiling for the current GPTQ bring-up configs so baseline validation can proceed with weight-only layer summaries first

Current status:

- a clean rerun of `R3_Q17B` is underway / pending trustworthy artifacts
- until that rerun lands, do not use the current `R3_Q17B` result as the GPTQ anchor for transfer experiments

### GPTQ Modal Bring-up Follow-up

- Date: 2026-03-13
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `GPTQ 4-bit`
- Scope: Modal bring-up debugging, not yet a valid transfer result
- Detailed status: `docs/experiments/qwen3_1p7b_gptq_bringup_status.md`

What changed during bring-up:

- forced GPTQ runs toward `float16`
- split the GPTQ baseline into a reference FP model and a separate source model to avoid profiling from a mutated quantized model
- changed the Modal GPTQ wrapper to:
  - persist artifacts to a Modal volume
  - support detached execution
  - invoke the remote function directly

What was learned:

- the original completed baseline is numerically unusable
- the corrected config reaches the live container
- but the corrected full baseline still is not landing as a trustworthy committed artifact

Current decision:

- do not proceed to `G2B02_Q17B` / `G2R02_Q17B`
- treat GPTQ as still in bring-up / debugging state
- next debugging step should be a much smaller GPTQ smoke baseline before another full `R3_Q17B`

### GPTQ Pause Point

- Date: 2026-03-13
- Scope: pause / handoff note

Final state before pausing:

- the most recent `R3_Q17B` Modal wrapper remained alive for hours in a sleeping state
- the local artifact timestamps refreshed, but the content still matched the old broken baseline
- no trustworthy GPTQ baseline was produced

What is safe to do next time:

1. check the Modal dashboard for any stale live GPTQ app and stop it
2. do not trust the current `results/modal/qwen3_1p7b_gptq_baselines/R3_Q17B` metrics as valid
3. add a smaller `R3S_Q17B` smoke config with no profiling
4. validate sane perplexity first
5. only then retry full `R3_Q17B`
6. only after a valid baseline exists, resume `G2B02_Q17B` and `G2R02_Q17B`

### Kaggle RTN Reproduction Check

- Date: 2026-03-13
- Model: `Qwen/Qwen3-1.7B-Base`
- Method family: `RTN` transfer reruns on Kaggle
- Scope: reproduction / robustness check

Observed Kaggle metrics:

- `P2B02_Q17B`: `21.2432`
- `P2R02_Q17B`: `21.2982`
- `P2B03_Q17B`: `21.1469`
- `P2R03_Q17B`: `21.2982`

Interpretation:

- Kaggle reproduces the same ordering already seen on Modal
- targeted bits still beat targeted rank on the `1.7B` RTN transfer study
- Phase 2 can be considered closed with a stronger environment-level confirmation

### Kaggle GPTQ Smoke Checks

- Date: 2026-03-14
- Model: `Qwen/Qwen3-1.7B-Base`
- Method family: `GPTQ` smoke debugging on Kaggle
- Scope: backend validation, not transfer

Observed Kaggle smoke results:

- `R3S_Q17B`: `perplexity = NaN`, `dtype = float16`
- `R3S2_Q17B`: `perplexity = NaN`, `dtype = float16`

Interpretation:

- Kaggle got the GPTQ path past the earlier install/build barrier
- but the quantized model is still numerically unstable at evaluation time
- so GPTQ remains blocked even after a working Kaggle environment

Decision:

- GPTQ remained blocked at that point and should not have been treated as ready for transfer

### Modal GPTQ Recovery and Transfer Completion

- Date: 2026-03-14
- Model: `Qwen/Qwen3-1.7B-Base`
- Platform: `Modal`
- Scope: GPTQ recovery, baseline validation, and first matched transfer frontier

What changed:

- patched the Transformers GPTQ path to set `hf_device_map` before Optimum packing
- added explicit finite-logit validation before perplexity evaluation
- changed GPTQ targeted updates to replace selected packed modules with floating `nn.Linear` modules instead of trying to overwrite packed tensors directly

Validated GPTQ runs:

- `R3_Q17B`
  - perplexity: `15.9137`
  - memory: `1356968233`
- `G2B02_Q17B`
  - perplexity: `15.8993`
  - memory: `1364308265`
  - upgraded layers: `7`
- `G2R02_Q17B`
  - perplexity: `15.8823`
  - memory: `1360965929`
  - repair bytes: `3997696`
- `G2B03_Q17B`
  - perplexity: `15.8914`
  - memory: `1383182633`
  - upgraded layers: `10`
- `G2R03_Q17B`
  - perplexity: `15.8823`
  - memory: `1360965929`
  - repair bytes: `3997696`

Interpretation:

- GPTQ no longer has a bring-up blocker on Modal
- the first valid GPTQ transfer point favors targeted rank over targeted bits
- the current GPTQ rank action space saturates the candidate pool by `+1.0%`
- because of that, `G2R03_Q17B` does not improve over `G2R02_Q17B`
- this means the next GPTQ improvement, if we keep going, is not “run more points”
- it is “expand or refine the rank/bit action space so the frontier can keep spending budget meaningfully”

- do not run full `R3_Q17B`
- do not run `G2B02_Q17B` / `G2R02_Q17B`
- treat GPTQ as paused again until the backend path is changed or more deeply debugged

### Run R3

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `GPTQ 4-bit`
- Script: `./scripts/run_r3_gptq_4bit.sh`
- Log: `results/logs/phase1_20260311_142251.log`
- Metrics target: `results/phase1/R3/metrics.json`

Command:

```bash
./scripts/run_r3_gptq_4bit.sh
```

Result:

- status: `blocked`
- blocker: missing Python package `optimum`

Observed behavior:

- the generic harness prepared the run correctly
- execution did not start because the GPTQ backend dependency check failed immediately

Interpretation:

- this is not a model or evaluation failure
- it is only an environment dependency issue for the GPTQ path
- `R3` does not yet contribute empirical results, but it does confirm that the dependency guard is working as intended

Next action:

1. install GPTQ-supporting dependencies in `rl`
2. rerun `R3`
3. compare GPTQ residual structure against RTN residual structure

## Phase 2

### Run P2B01

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `targeted_mixed_precision`
- Script: `./scripts/run_p2b01_bits_uniform_0p25pct.sh`
- Log: `results/logs/phase2_20260311_211918.log`
- Metrics: `results/phase2/P2B01/metrics.json`
- Actions: `results/phase2/P2B01/actions.json`

Command:

```bash
./scripts/run_p2b01_bits_uniform_0p25pct.sh
```

Result:

- status: `completed`
- allocator: `uniform_bits`
- extra budget bytes: `768261`
- selected actions: `[]`
- perplexity: `30.516916135923783`
- memory total bytes: `307304448`

Interpretation:

- the run executed correctly
- no matrix-level action fit inside the `+0.25%` budget
- this exactly reproduced the `R2` baseline

Why this matters:

- the scaffold and accounting are behaving sensibly
- matrix-level actions are still too coarse at this budget
- Phase 2 needs either larger budget points or finer granularity for early frontier points

### Run P2B02

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `targeted_mixed_precision`
- Script: `./scripts/run_p2b02_bits_activation_1p0pct.sh`
- Log: `results/logs/phase2_20260311_213655.log`
- Metrics: `results/phase2/P2B02/metrics.json`
- Actions: `results/phase2/P2B02/actions.json`

Command:

```bash
./scripts/run_p2b02_bits_activation_1p0pct.sh
```

Result:

- status: `completed`
- allocator: `greedy_activation`
- extra budget bytes: `3073044`
- selected actions:
  - `model.layers.14.self_attn.o_proj`
  - `model.layers.13.self_attn.o_proj`
- perplexity: `30.42377952396118`
- memory total bytes: `309401600`

Comparison vs `R2`:

- `R2` perplexity: `30.5169`
- `P2B02` perplexity: `30.4238`
- improvement: about `0.0931`

Interpretation:

- this is the first meaningful Phase 2 bits-only frontier point
- the activation-greedy allocator chose two `self_attn.o_proj` matrices because their gain-per-byte score beat the more expensive `mlp.down_proj` options
- that is a useful result even though the absolute improvement is still modest

Immediate takeaway:

- matrix-level targeted bits are now working
- the first selected actions are not the highest raw-damage layers, but the best gain-per-byte layers under the current cost model
- this already shows why Phase 2 needed a frontier view instead of simple top-damage ranking

### Run P2R02

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `targeted_svd_rank`
- Script: `./scripts/run_p2r02_rank_activation_1p0pct.sh`
- Log: `results/logs/phase2_20260311_221537.log`
- Metrics: `results/phase2/P2R02/metrics.json`

## Qwen3 1.7B Modal Scale-Up

### Run R2_Q17B

- Date: 2026-03-12
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `RTN 4-bit`
- Execution path: `Modal`
- Script: `./scripts/run_qwen3_1p7b_r2_modal.sh`
- Metrics: `results/modal/qwen3_1p7b_baselines/R2_Q17B/metrics.json`
- Layer errors: `results/modal/qwen3_1p7b_baselines/R2_Q17B/layer_errors.json`

Result:

- status: `completed`
- perplexity: `21.310207660816054`
- memory total bytes: `887107584`
- latency ms/token: `1.663287446559387`

Interpretation:

- the `1.7B` RTN baseline completed cleanly on Modal
- this produced the first larger-model anchor point for the transfer study
- the baseline layer errors were used to build a `1.7B`-specific top-12 candidate pool instead of reusing the `0.6B` pool

Action taken:

- built `configs/scaleup_1p7b/qwen3_1p7b_r2_top12_candidate_pool.json` from the Modal-synced layer error file
- switched the transfer runs to use `A10G` as the default practical GPU target

### Run P2B02_Q17B

- Date: 2026-03-12
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_mixed_precision`
- Execution path: `Modal`
- Script: `MODAL_GPU=A10G ./scripts/run_qwen3_1p7b_p2b02_modal.sh`
- Metrics: `results/modal/qwen3_1p7b_transfer/P2B02_Q17B/metrics.json`
- Actions: `results/modal/qwen3_1p7b_transfer/P2B02_Q17B/actions.json`

Result:

- status: `completed`
- extra budget bytes: `8871076`
- perplexity: `21.24151620487837`
- memory total bytes: `895496192`

Selected actions:

- `model.layers.16.self_attn.o_proj`
- `model.layers.14.self_attn.o_proj`
- `model.layers.13.self_attn.o_proj`
- `model.layers.15.self_attn.o_proj`

Interpretation:

- this is the first valid `1.7B` bits-only transfer point
- targeted bits improved over the `R2_Q17B` baseline by about `0.0687` perplexity
- the allocator again preferred `self_attn.o_proj` matrices under the current matrix-level cost model

### Run P2R02_Q17B

- Date: 2026-03-12
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_svd_rank`
- Execution path: `Modal`
- Script: `MODAL_GPU=A10G ./scripts/run_qwen3_1p7b_p2r02_modal.sh`
- Metrics: `results/modal/qwen3_1p7b_transfer/P2R02_Q17B/metrics.json`
- Actions: `results/modal/qwen3_1p7b_transfer/P2R02_Q17B/actions.json`

Result:

- status: `completed`
- extra budget bytes: `8871076`
- repair factor bytes: `4456448`
- perplexity: `21.300059072861647`
- memory total bytes: `891564032`

Interpretation:

- targeted rank also improved over the `R2_Q17B` baseline, but only slightly
- at the current `+1.0%` budget point on `1.7B`, targeted bits beat targeted rank
- this is the first clear sign that the `0.6B` local Phase 2 result does not transfer directly to the larger model

Current takeaway:

- `0.6B` under local RTN favored targeted rank
- `1.7B` on Modal currently favors targeted bits at `+1.0%`
- the next required step is the `+2.0%` pair on `1.7B`:
  - `P2B03_Q17B`
  - `P2R03_Q17B`
- that pair is needed to decide whether the scale-up result is stable or only a one-budget effect
- Actions: `results/phase2/P2R02/actions.json`

Command:

```bash
./scripts/run_p2r02_rank_activation_1p0pct.sh
```

Result after allocator fix:

- status: `completed`
- allocator: `greedy_activation`
- extra budget target: `3073044` bytes
- actual repair factor bytes: `3014656`
- selected actions: `54` incremental rank actions
- perplexity: `29.72228563777474`
- memory total bytes: `310319104`

Comparison:

- `R2` perplexity: `30.5169`
- `P2B02` perplexity: `30.4238`
- `P2R02` perplexity: `29.7223`

Interpretation:

- the first `P2R02` run was not fair because the old rank allocator under-spent the budget by choosing only one tiny rank action per layer
- after changing the action space to incremental rank chunks, the rerun consumed almost the full intended budget
- under this corrected comparison, targeted rank clearly beat the `P2B02` bits-only point at the same nominal budget scale

What the allocator did:

- it first spread `rank-4` repairs across a set of high-value `mlp.down_proj` and `self_attn.o_proj` matrices
- then it continued adding extra rank to the strongest matrices through `4 -> 8`, `8 -> 16`, and later steps
- this is much closer to the intended Phase 2 frontier behavior

Immediate consequence:

- the Phase 2 story is now genuinely open again
- the current working conclusion is no longer "bits first" by default
- instead, the better statement is:
  - at least at the `+1.0%` frontier point under RTN, targeted rank can outperform the current matrix-level bits-only allocator

Next question:

- does this advantage persist at `+2.0%`, or does bits catch up once larger matrix-level upgrades fit more naturally?

### Run P2B03

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `targeted_mixed_precision`
- Script: `./scripts/run_p2b03_bits_activation_2p0pct.sh`
- Log: `results/logs/phase2_20260311_223722.log`
- Metrics: `results/phase2/P2B03/metrics.json`
- Actions: `results/phase2/P2B03/actions.json`

Command:

```bash
./scripts/run_p2b03_bits_activation_2p0pct.sh
```

Result:

- status: `completed`
- allocator: `greedy_activation`
- extra budget bytes: `6146089`
- selected actions: `5`
- perplexity: `30.250561361308314`
- memory total bytes: `312547328`

Selected layer types:

- all selected upgrades were `self_attn.o_proj`

Interpretation:

- the bits-only frontier improved versus `P2B02`
- but it stayed locked onto attention output projections
- even at `+2.0%`, it still did not catch the corrected `P2R02` result

### Run P2R03

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `targeted_svd_rank`
- Script: `./scripts/run_p2r03_rank_activation_2p0pct.sh`
- Log: `results/logs/phase2_20260311_225223.log`
- Metrics: `results/phase2/P2R03/metrics.json`
- Actions: `results/phase2/P2R03/actions.json`

Command:

```bash
./scripts/run_p2r03_rank_activation_2p0pct.sh
```

Result:

- status: `completed`
- allocator: `greedy_activation`
- extra budget target: `6146089` bytes
- actual repair factor bytes: `3211264`
- selected actions: `56` incremental rank actions
- perplexity: `29.725154398491505`
- memory total bytes: `310515712`

Comparison:

- `P2B03` perplexity: `30.2506`
- `P2R03` perplexity: `29.7252`

Interpretation:

- targeted rank still beat targeted bits at the larger budget point
- the rank frontier improved only marginally relative to `P2R02`, which suggests diminishing returns or candidate-pool saturation
- this was enough to close the local RTN Phase 2 question without requiring a hybrid second-stage detour

### GPTQ Dependency Install Attempt

- Date: 2026-03-11
- Script: `./scripts/install_gptq_deps_rl.sh`
- Log: `results/logs/install_gptq_deps_20260311_142641.log`

Command:

```bash
./scripts/install_gptq_deps_rl.sh
```

Result:

- status: `failed`
- primary failure: `gptqmodel` wheel build failed

Actual cause:

- the build tried to compile CUDA kernels that require `sm_80` or higher
- the machine GPU is `NVIDIA GeForce GTX 1660 SUPER`, which is `sm_75`
- the build failed with errors like:
  - `Feature 'cp.async' requires .target sm_80 or higher`
  - `Feature '.m16n8k16' requires .target sm_80 or higher`

Environment side effects:

- `torch` remained at `2.6.0+cu124`
- `transformers` changed from `4.56.2` to `5.3.0`
- `accelerate` changed from `1.4.0` to `1.13.0`
- `optimum` is now installed
- `gptqmodel` is not installed

Interpretation:

- this was not just a missing-package issue
- the attempted GPTQ dependency path is not compatible with the current GPU architecture
- the install script was too aggressive because it upgraded core packages in a working env

Immediate caution:

- do not assume the `rl` env is unchanged
- rerun a small smoke test before further experiments

Next action:

1. verify that `R1` and `R2` paths still run after the package changes
2. avoid `gptqmodel` on this GPU unless a known `sm_75` compatible path is identified
3. prefer either:
   - a different GPTQ implementation path compatible with older GPUs, or
   - proceed with RTN / mixed-precision / low-rank experiments first

### Run R4

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `uniform_svd_repair`
- Script: `./scripts/run_r4_uniform_svd_rank4.sh`
- Log: `results/logs/phase1_20260311_145205.log`
- Metrics: `results/phase1/R4/metrics.json`
- Layer errors: `results/phase1/R4/layer_errors.json`
- Residual profiles: `results/phase1/R4/residual_profiles.json`

Command:

```bash
./scripts/run_r4_uniform_svd_rank4.sh
```

Result:

- status: `completed`
- device: `cuda`
- dtype: `bfloat16`
- perplexity: `30.557722980280193`
- evaluated tokens: `300979`
- memory total bytes: `307550208`
- repair factor bytes: `245760`
- latency ms/token: `0.6853179166752205`

Comparison vs R2:

- `R2` perplexity: `30.5169`
- `R4` perplexity: `30.5577`
- `R2` memory: `307304448` bytes
- `R4` memory: `307550208` bytes

Observed effect:

- rank-4 uniform repair on the top 8 damaged layers did not improve perplexity
- it was slightly worse than the RTN baseline while using slightly more memory

Interpretation:

- this is a negative but useful result
- the simplest low-rank repair setting is too weak at rank 4 on the currently selected layers
- the project should not conclude from `R4` alone that low-rank repair fails
- the next meaningful checks are higher ranks and the equal-budget bits-only baseline

Immediate next action:

1. run `R5` and `R6`
2. compare whether higher repair rank changes the trend
3. then run `R8` to compare against equal-budget extra bits for the same budget scale

### Run R5

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `uniform_svd_repair`
- Script: `./scripts/run_r5_uniform_svd_rank8.sh`
- Log: `results/logs/phase1_20260311_155218.log`
- Metrics: `results/phase1/R5/metrics.json`
- Residual profiles: `results/phase1/R5/residual_profiles.json`

Command:

```bash
./scripts/run_r5_uniform_svd_rank8.sh
```

Result:

- status: `completed`
- device: `cuda`
- dtype: `bfloat16`
- perplexity: `30.46538282428384`
- evaluated tokens: `300979`
- memory total bytes: `307795968`
- repair factor bytes: `491520`
- latency ms/token: `0.6852693627330594`

Comparison:

- `R2` perplexity: `30.5169`
- `R4` perplexity: `30.5577`
- `R5` perplexity: `30.4654`

Observed effect:

- rank-8 uniform repair improved over both `R4` and the plain RTN baseline
- the improvement over `R2` is small but real

Interpretation:

- low-rank repair is not completely dead under RTN
- rank 4 was too weak, but rank 8 starts to recover some quality
- this keeps the repair direction alive
- the next key question is whether this gain is better or worse than spending the same extra bytes on more bits

Immediate next action:

1. run `R8` to compare against the equal-budget bits-only baseline matched to `R4`
2. run `R6` to see whether higher repair rank strengthens the recovery trend
3. compare repair gains per added byte across `R4`, `R5`, and the bits-only baselines

### Run R8

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `mixed_precision_budget_match`
- Script: `./scripts/run_r8_bits_match_r4.sh`
- Log: `results/logs/phase1_20260311_161300.log`
- Metrics: `results/phase1/R8/metrics.json`
- Layer errors: `results/phase1/R8/layer_errors.json`

Command:

```bash
./scripts/run_r8_bits_match_r4.sh
```

Result:

- status: `completed`
- perplexity: `30.516916135923783`
- memory total bytes: `307304448`
- upgraded layers: `[]`
- target memory match run id: `R4`

Observed effect:

- `R8` exactly reproduced the `R2` baseline
- no layers were upgraded

Interpretation:

- the equal-budget bits-only comparator is currently too coarse at the `R4` budget scale
- the extra bytes introduced by `R4` were not enough to upgrade even one selected layer from 4-bit to 8-bit
- this means `R8` is not yet a meaningful competitive baseline; it is effectively the unchanged RTN baseline

Practical implication:

- `R4` vs `R8` does not tell us whether rank-4 repair is better than extra bits
- it only tells us that the first matched budget is too small for the current bits-only upgrade granularity

Immediate next action:

1. run `R6` and `R10`
2. check whether a larger repair budget is finally large enough to upgrade at least one layer in the bits-only baseline
3. if not, refine the bits-only comparator granularity before relying on those comparisons

### Run R6

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `uniform_svd_repair`
- Script: `./scripts/run_r6_uniform_svd_rank16.sh`
- Metrics: `results/phase1/R6/metrics.json`
- Residual profiles: `results/phase1/R6/residual_profiles.json`

Result:

- status: `completed`
- perplexity: `30.315506091916657`
- memory total bytes: `308287488`
- repair factor bytes: `983040`

Comparison:

- `R2`: `30.5169`
- `R5`: `30.4654`
- `R6`: `30.3155`

Interpretation:

- the repair trend continues to improve as rank increases
- rank 16 is clearly better than rank 8 in this setup
- uniform low-rank repair is now recovering a non-trivial portion of the RTN loss

### Run R10

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `mixed_precision_budget_match`
- Script: `./scripts/run_r10_bits_match_r6.sh`
- Metrics: `results/phase1/R10/metrics.json`
- Layer errors: `results/phase1/R10/layer_errors.json`

Result:

- status: `completed`
- perplexity: `30.516916135923783`
- memory total bytes: `307304448`
- upgraded layers: `[]`
- target memory match run id: `R6`

Interpretation:

- even at the `R6` budget scale, the current bits-only comparator still cannot upgrade one full selected layer
- `R10` again collapses to the unchanged RTN baseline
- this confirms that the current bits-only comparison is still too coarse for the `R4-R6` budget range

Immediate next action:

1. run `R7`
2. run `R11`
3. check whether the `R7` budget is finally large enough to produce a non-empty bits-only upgrade set

### Run R7

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `uniform_svd_repair`
- Script: `./scripts/run_r7_uniform_svd_rank32.sh`
- Metrics: `results/phase1/R7/metrics.json`
- Residual profiles: `results/phase1/R7/residual_profiles.json`

Result:

- status: `completed`
- perplexity: `29.984764915103966`
- memory total bytes: `309270528`
- repair factor bytes: `1966080`

Comparison:

- `R2`: `30.5169`
- `R5`: `30.4654`
- `R6`: `30.3155`
- `R7`: `29.9848`

Interpretation:

- the repair trend continues monotonically as rank increases
- rank 32 is the best uniform-repair result so far
- uniform low-rank correction does recover a meaningful amount of quality under RTN

### Run R11

- Date: 2026-03-11
- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `mixed_precision_budget_match`
- Script: `./scripts/run_r11_bits_match_r7.sh`
- Metrics: `results/phase1/R11/metrics.json`
- Layer errors: `results/phase1/R11/layer_errors.json`

Result:

- status: `completed`
- perplexity: `28.861768021353583`
- memory total bytes: `308877312`
- upgraded layers:
  - `model.layers.27.mlp.down_proj`
- target memory match run id: `R7`

Interpretation:

- this is the first meaningful bits-only comparison
- at the `R7` budget scale, the bits-only baseline can finally upgrade one full selected layer
- and it beats the rank-32 repair baseline by a large margin

Working conclusion:

- low-rank repair is helpful, but the first real equal-budget bits-only baseline wins clearly in this setup
- this shifts the project direction:
  - mixed-precision allocation now looks stronger than uniform low-rank repair as the mainline story
  - low-rank repair may still matter as a secondary method or in combination with bit allocation, but it is no longer the leading baseline on the current evidence

Immediate next action:

1. generate a consolidated summary table for `R1-R11`
2. write a short decision memo saying the current evidence favors mixed precision over uniform low-rank repair
3. decide whether to:
   - refine the bits-only allocator,
   - or test targeted low-rank repair rather than uniform repair

## Documentation Rule Going Forward

Every run should add one new section here with:

- date
- command used
- log path
- output file paths
- key metrics
- main observations
- interpretation
- next action triggered by the result

## Qwen3 1.7B Modal Scale-Up

### Run R2_Q17B

- Date: 2026-03-12
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `RTN 4-bit`
- Execution path: `Modal`
- GPU: `T4`
- Script: `./scripts/run_qwen3_1p7b_r2_modal.sh`
- Metrics: `results/modal/qwen3_1p7b_baselines/R2_Q17B/metrics.json`
- Layer errors: `results/modal/qwen3_1p7b_baselines/R2_Q17B/layer_errors.json`
- Log: `results/modal/qwen3_1p7b_baselines/R2_Q17B/modal_run.log`

Result:

- status: `completed`
- device: `cuda`
- dtype: `bfloat16`
- perplexity: `21.310207660816054`
- evaluated tokens: `300979`
- memory total bytes: `887107584`
- memory metadata bytes: `26882048`
- latency ms/token: `1.663287446559387`

Main observations:

- the `1.7B` baseline completed successfully on Modal and is now durable on disk
- the top activation-damage layers are concentrated in mid-stack `mlp.down_proj` and `self_attn.o_proj`
- the shared top-12 candidate pool was built from the Modal-synced `layer_errors.json`

Interpretation:

- the scale-up path is working
- the project now has a valid `1.7B` anchor baseline for matched transfer runs
- `T4` is viable for the baseline, but it is slow enough that `A10G` is the better default for repeated transfer experiments

Next action:

1. rerun `P2B02_Q17B` cleanly on `A10G`
2. run `P2R02_Q17B` on `A10G`
3. compare the first matched `1.7B` bits-vs-rank frontier point

### Run P2B02_Q17B

- Date: 2026-03-12
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_mixed_precision`
- Execution path: `Modal`
- Intended GPU: `A10G`
- Script: `./scripts/run_qwen3_1p7b_p2b02_modal.sh`
- Metrics target: `results/modal/qwen3_1p7b_transfer/P2B02_Q17B/metrics.json`

Current state:

- not a valid completed result yet
- the repo currently contains only prepared or partially synced placeholder files for this run

Why it is incomplete:

- earlier attempts were interrupted before the local Modal wrapper finished writing back the remote payload
- this left `metrics.json` with `status: pending` and no usable allocator output

Interpretation:

- do not treat `P2B02_Q17B` as a real experimental result yet
- the run needs one clean uninterrupted completion before any comparison is valid

Next action:

1. rerun `P2B02_Q17B` on `A10G`
2. verify `metrics.json` and `actions.json`
3. then run `P2R02_Q17B` on `A10G`

### Run P2B03_Q17B

- Date: 2026-03-12
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_mixed_precision`
- Execution path: `Modal`
- Initial GPU: `A10G`
- Recovery GPU: `A100`
- Script: `./scripts/run_qwen3_1p7b_p2b03_modal.sh`
- Metrics target: `results/modal/qwen3_1p7b_transfer/P2B03_Q17B/metrics.json`

Result:

- status: `completed`
- extra budget bytes: `17742152`
- perplexity: `21.15047099410901`
- memory total bytes: `901787648`
- selected matrix upgrades: `7`

Observed behavior:

- the `A10G` app remained `ephemeral` for nearly one hour without local artifact sync
- Modal logs did not yield useful progress output for diagnosis
- this materially exceeded the earlier `1.7B` transfer runtimes for `P2B02_Q17B` and `P2R02_Q17B`

Interpretation:

- the original `A10G` attempt was not reliable enough to keep
- the recovery rerun on `A100` completed and produced the valid `+2.0%` bits result
- the larger budget improved over `P2B02_Q17B`, so the `1.7B` bits frontier is still moving in the right direction

Next action:

1. compare against `P2R03_Q17B`
2. finalize the `1.7B` transfer conclusion

### Run P2R03_Q17B

- Date: 2026-03-13
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_svd_rank`
- Execution path: `Modal`
- GPU: `A100`
- Script: `./scripts/run_qwen3_1p7b_p2r03_modal.sh`
- Metrics: `results/modal/qwen3_1p7b_transfer/P2R03_Q17B/metrics.json`
- Actions: `results/modal/qwen3_1p7b_transfer/P2R03_Q17B/actions.json`

Result:

- status: `completed`
- extra budget bytes: `17742152`
- perplexity: `21.297133858312165`
- memory total bytes: `891564032`
- repair factor bytes: `4456448`

Interpretation:

- targeted rank did not materially improve beyond the earlier `P2R02_Q17B` point
- at the same `+2.0%` transfer point, targeted bits beat targeted rank again
- this confirms that the `1.7B` transfer result is stable across the tested budget range

Transfer conclusion:

- on `Qwen/Qwen3-1.7B-Base` under the current `RTN` + matrix-level action setup, targeted bits beat targeted rank at both `+1.0%` and `+2.0%`
- this differs from the earlier `0.6B` local result, which favored targeted rank
- the project now has a model-scale-dependent result instead of a single universal winner

### GPTQ Modal Bring-up

- Date: 2026-03-13
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `GPTQ 4-bit`
- Execution path: `Modal`
- Script: `./scripts/run_qwen3_1p7b_r3_gptq_modal.sh`

Bring-up notes:

- first Modal image attempt failed while building `gptqmodel` in the same install step as `torch`
- fixing that required:
  - moving `gptqmodel` into a separate install step
  - upgrading the GPTQ Modal image to `torch==2.7.1`
- second image attempt then failed during `gptqmodel` metadata generation because `bdist_wheel` was unavailable
- the current image now installs `pip`, `setuptools`, and `wheel` before the GPTQ step

Current state:

- the GPTQ baseline code path is implemented locally
- the dedicated Modal GPTQ runner is implemented
- the latest `R3_Q17B` Modal attempt is now past the original packaging blockers and is rebuilding with the corrected image

### GPTQ Smoke Bring-up

- Date: 2026-03-13
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `GPTQ 4-bit`
- Execution path: `Modal`
- GPU: `A100`
- Script: `./scripts/run_qwen3_1p7b_r3_gptq_smoke_modal.sh`
- Run id: `R3S_Q17B`
- Modal app: `ap-72mSaICHTsZxrQmK2fF4CK`

Bring-up changes:

- fixed the detached GPTQ Modal path so the remote run writes into the mounted results volume
- added a smaller smoke baseline with:
  - `16` calibration texts
  - `16` evaluation sequences
  - no activation profiling
  - no residual profiling

Live status at last check:

- the mounted run directory exists at `results/modal/qwen3_1p7b_gptq_smoke/R3S_Q17B`
- `metrics.json` is present there in `pending` state
- the Modal app is still live on `A100`
- GPU memory is allocated and the run appears to be compute-bound rather than crashing immediately

Interpretation:

- the results-persistence bug is fixed
- GPTQ transfer is still blocked until this smoke run either:
  - finishes with sane perplexity, or
  - exposes the next concrete backend/runtime failure

## Phase 3

### Run R2_S3B

- Date: 2026-03-14
- Model: `HuggingFaceTB/SmolLM3-3B-Base`
- Method: `RTN 4-bit`
- Execution path: `Modal`
- GPU: `A10G`
- Script: `./scripts/run_smollm3_3b_r2_modal.sh`
- Metrics: `results/modal/smollm3_3b_baselines/R2_S3B/metrics.json`

Result:

- status: `completed`
- perplexity: `47.91685627870016`
- memory total bytes: `1585520640`
- memory metadata bytes: `48046080`
- latency ms/token: `0.12940787724541597`

Notes:

- this is the Phase 3 bridge-scale baseline
- the RTN path needed a vectorized quantization implementation before this model became practical on `A10G`
- after that optimization, baseline execution was stable and cheap enough to continue on the smaller GPU

Interpretation:

- the `3B` scale point is now anchored
- candidate-pool construction and the first matched frontier pair were justified

### Run P3B02_S3B

- Date: 2026-03-14
- Model: `HuggingFaceTB/SmolLM3-3B-Base`
- Method: `targeted_mixed_precision`
- Execution path: `Modal`
- GPU: `A10G`
- Script: `./scripts/run_smollm3_3b_p3b02_modal.sh`
- Metrics: `results/modal/smollm3_3b_transfer/P3B02_S3B/metrics.json`
- Actions: `results/modal/smollm3_3b_transfer/P3B02_S3B/actions.json`

Result:

- status: `completed`
- extra budget bytes: `15855206`
- perplexity: `47.4955338117833`
- memory total bytes: `1591812096`

Selected actions:

- `model.layers.7.self_attn.o_proj` `4 -> 8`
- `model.layers.3.self_attn.o_proj` `4 -> 8`
- `model.layers.0.self_attn.o_proj` `4 -> 8`

Interpretation:

- targeted bits improved over the `R2_S3B` baseline at the first matched budget point
- the gain-per-byte policy preferred `self_attn.o_proj` matrices over the higher-damage `mlp.down_proj` matrices

### Run P3R02_S3B

- Date: 2026-03-14
- Model: `HuggingFaceTB/SmolLM3-3B-Base`
- Method: `targeted_svd_rank`
- Execution path: `Modal`
- GPU: `A10G`
- Script: `./scripts/run_smollm3_3b_p3r02_modal.sh`
- Metrics: `results/modal/smollm3_3b_transfer/P3R02_S3B/metrics.json`
- Actions: `results/modal/smollm3_3b_transfer/P3R02_S3B/actions.json`

Result:

- status: `completed`
- extra budget bytes: `15855206`
- perplexity: `47.98327419715927`
- memory total bytes: `1595498496`
- repair factor bytes: `9977856`

Interpretation:

- targeted rank was slightly worse than the `R2_S3B` baseline and clearly worse than `P3B02_S3B`
- this made the first matched pair decisive enough to skip the `+2.0%` pair for `SmolLM3-3B`
- the next cost-effective move is to proceed directly to the `Qwen/Qwen3-8B-Base` validation-scale point

### Run R2_Q8B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-8B-Base`
- Method: `RTN 4-bit`
- Execution path: `Modal`
- GPU: `A100`
- Script: `./scripts/run_qwen3_8b_r2_modal.sh`
- Metrics: `results/modal/qwen3_8b_baselines/R2_Q8B/metrics.json`

Result:

- status: `completed`
- perplexity: `16.193944972311687`
- memory total bytes: `3902300160`
- memory metadata bytes: `118251520`
- latency ms/token: `0.12391592523730807`

Implementation note:

- the first `8B` attempts exposed a large-model limitation in the RTN path
- the fix was:
  - CPU-side RTN working tensors during quantization
  - sequential model offload during activation profiling

Interpretation:

- with those changes, the `8B` baseline became practical on `A100`
- this was enough to complete the validation-scale Phase 3 point without escalating hardware further

### Run P3B02_Q8B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-8B-Base`
- Method: `targeted_mixed_precision`
- Execution path: `Modal`
- GPU: `A100`
- Script: `./scripts/run_qwen3_8b_p3b02_modal.sh`
- Metrics: `results/modal/qwen3_8b_transfer/P3B02_Q8B/metrics.json`
- Actions: `results/modal/qwen3_8b_transfer/P3B02_Q8B/actions.json`

Result:

- status: `completed`
- extra budget bytes: `39023002`
- perplexity: `16.142918191325858`
- memory total bytes: `3935854592`

Selected actions:

- `model.layers.21.self_attn.o_proj` `4 -> 8`
- `model.layers.20.self_attn.o_proj` `4 -> 8`
- `model.layers.18.self_attn.o_proj` `4 -> 8`
- `model.layers.22.self_attn.o_proj` `4 -> 8`

Interpretation:

- targeted bits improved over the `R2_Q8B` baseline at the first matched budget point
- the action pattern is consistent with the `1.7B` and `3B` results: late `self_attn.o_proj` matrices remain the highest-value bit-upgrade targets

### Run P3R02_Q8B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-8B-Base`
- Method: `targeted_svd_rank`
- Execution path: `Modal`
- GPU: `A100`
- Script: `./scripts/run_qwen3_8b_p3r02_modal.sh`
- Metrics: `results/modal/qwen3_8b_transfer/P3R02_Q8B/metrics.json`
- Actions: `results/modal/qwen3_8b_transfer/P3R02_Q8B/actions.json`

Result:

- status: `completed`
- extra budget bytes: `39023002`
- perplexity: `16.20352045227616`
- memory total bytes: `3911213056`
- repair factor bytes: `8912896`

Interpretation:

- targeted rank was slightly worse than the `R2_Q8B` baseline and clearly worse than `P3B02_Q8B`
- this made the first matched pair decisive enough to skip the `+2.0%` pair for `Qwen3-8B`
- Phase 3 `RTN` now has a stable cross-scale conclusion:
  - `0.6B` favored targeted rank
  - `1.7B`, `3B`, and `8B` favored targeted bits

### Run R3_S3B

- Date: 2026-03-14
- Model: `HuggingFaceTB/SmolLM3-3B-Base`
- Method: `gptq`
- Execution path: `Modal`
- GPU: `A10G`
- Script: `./scripts/run_smollm3_3b_r3_gptq_modal.sh`
- Metrics: `results/modal/smollm3_3b_gptq_baselines/R3_S3B/metrics.json`
- Validation: `results/modal/smollm3_3b_gptq_baselines/R3_S3B/gptq_validation.json`

Result:

- status: `completed`
- perplexity: `11.536600075046657`
- memory total bytes: `1990237781`
- latency ms/token: `0.1710847020350192`
- finite output validation: `passed`

Interpretation:

- the `3B` GPTQ baseline is valid on Modal
- `A10G` is sufficient for the `3B` GPTQ baseline

### Run G3B02_S3B

- Date: 2026-03-14
- Model: `HuggingFaceTB/SmolLM3-3B-Base`
- Method: `targeted_mixed_precision`
- Execution path: `Modal`
- GPU: `A100`
- Script: `./scripts/run_smollm3_3b_g3b02_modal.sh`
- Metrics: `results/modal/smollm3_3b_gptq_transfer/G3B02_S3B/metrics.json`
- Actions: `results/modal/smollm3_3b_gptq_transfer/G3B02_S3B/actions.json`

Result:

- status: `completed`
- extra budget bytes: `19902378`
- perplexity: `11.548343137492862`
- memory total bytes: `1997577813`

Implementation note:

- the first `A10G` attempt failed with a real CUDA OOM during GPTQ transfer-layer handling
- `A100` was required for the `3B` GPTQ transfer path under the current implementation

Interpretation:

- targeted bits regressed slightly relative to the `R3_S3B` GPTQ baseline
- selected actions concentrated in attention projection matrices, especially `v_proj` and `k_proj`

### Run G3R02_S3B

- Date: 2026-03-14
- Model: `HuggingFaceTB/SmolLM3-3B-Base`
- Method: `targeted_svd_rank`
- Execution path: `Modal`
- GPU: `A100`
- Script: `./scripts/run_smollm3_3b_g3r02_modal.sh`
- Metrics: `results/modal/smollm3_3b_gptq_transfer/G3R02_S3B/metrics.json`
- Actions: `results/modal/smollm3_3b_gptq_transfer/G3R02_S3B/actions.json`

Result:

- status: `completed`
- extra budget bytes: `19902378`
- perplexity: `11.64821293633937`
- memory total bytes: `2000477781`
- repair factor bytes: `10240000`

Interpretation:

- targeted rank regressed more strongly than targeted bits relative to the `R3_S3B` GPTQ baseline
- the `1.7B` GPTQ rank win does not transfer cleanly to `3B`
- the first matched `3B` GPTQ pair is already decisive enough that a `+2.0%` pair is not required before moving on to `8B`

### Run R3S_Q8B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-8B-Base`
- Method: `gptq`
- Execution path: `Modal`
- GPU: `A100`
- Script: `./scripts/run_qwen3_8b_r3_gptq_smoke_modal.sh`
- Metrics dir: `results/modal/qwen3_8b_gptq_smoke/R3S_Q8B`

Result:

- status: `blocked`
- no valid perplexity produced
- no valid `gptq_validation.json` produced

Observed failure modes:

1. Single-device placement (`device_map: "single"`)
- eliminated the earlier `accelerate` hook failure
- but failed with real CUDA OOM on `A100`

2. Auto placement (`device_map: "auto"`)
- avoided the single-device OOM
- but still failed during GPTQ quantization with `StopIteration` in the `accelerate` offload hook for a parameterless module (`rotary_emb`)

Artifacts:

- payload: `results/modal/qwen3_8b_gptq_smoke/R3S_Q8B/modal_payload.json`
- log: `results/modal/qwen3_8b_gptq_smoke/R3S_Q8B/modal_run.log`
- validation plan: `docs/roadmap/qwen3_8b_gptq_validation_plan.md`

Interpretation:

- `8B` GPTQ is not ready for full baseline or matched transfer runs under the current backend path
- the remaining issue is backend/runtime-specific, not just raw GPU size

### Run R3S_Q8B (A100-80GB Retry)

- Date: 2026-03-14
- Model: `Qwen/Qwen3-8B-Base`
- Method: `gptq`
- Execution path: `Modal`
- GPU: `A100-80GB`
- Script: `./scripts/run_qwen3_8b_r3_gptq_smoke_modal.sh`
- Metrics dir: `results/modal/qwen3_8b_gptq_smoke/R3S_Q8B.a10080_1773480462`

Result:

- status: `blocked`
- no valid perplexity produced
- no valid `gptq_validation.json` produced

Observed failure mode:

- `device_map: "auto"` still failed during GPTQ quantization with `StopIteration` in the `accelerate` offload hook for the parameterless `rotary_emb` path
- this reproduced even with `A100-80GB`, so the blocker is not resolved by moving from `40GB` to `80GB`

Artifacts:

- payload: `results/modal/qwen3_8b_gptq_smoke/R3S_Q8B.a10080_1773480462/modal_payload.json`
- log: `results/modal/qwen3_8b_gptq_smoke/R3S_Q8B.a10080_1773480462/modal_run.log`

Interpretation:

- `8B` GPTQ remains blocked after the `A100-80GB` retry
- the next step is backend/offload-path repair, not a larger GPU rerun

### Run R3S_Q8B (A100-80GB Single-Device Retry)

- Date: 2026-03-14
- Model: `Qwen/Qwen3-8B-Base`
- Method: `gptq`
- Execution path: `Modal`
- GPU: `A100-80GB`
- Script: `./scripts/run_qwen3_8b_r3_gptq_smoke_modal.sh`
- Metrics dir: `results/modal/qwen3_8b_gptq_smoke/R3S_Q8B`

Result:

- status: `completed`
- perplexity: `13.7503`
- memory: `6,103,927,291` bytes
- validation: finite logits and finite loss on the checked batch

What changed:

- the actual Modal GPU request was corrected to a real `A100-80GB`
- the config switched to `device_map: "single"` instead of `auto`

Interpretation:

- the `8B` GPTQ smoke path is valid on Modal
- the remaining blocker was the offload path, not the model itself

### Run R3_Q8B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-8B-Base`
- Method: `gptq`
- Execution path: `Modal`
- GPU: `A100-80GB`
- Script: `./scripts/run_qwen3_8b_r3_gptq_modal.sh`
- Metrics dir: `results/modal/qwen3_8b_gptq_baselines/R3_Q8B`

Result:

- status: `completed`
- perplexity: `11.7970`
- memory: `6,103,975,811` bytes
- validation: finite logits and finite loss on the checked batch

Interpretation:

- the full `8B` GPTQ baseline is now valid
- `Qwen3-8B` can use the same `single`-device strategy as the successful smoke run

### Run G2B02_Q8B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-8B-Base`
- Method: `targeted_mixed_precision`
- Base method: `gptq`
- Execution path: `Modal`
- GPU: `A100-80GB`
- Script: `./scripts/run_qwen3_8b_g2b02_gptq_modal.sh`
- Metrics dir: `results/modal/qwen3_8b_gptq_transfer/G2B02_Q8B`

Result:

- status: `completed`
- perplexity: `11.7823`
- memory: `6,150,113,155` bytes
- selected upgrades: `11`

Interpretation:

- targeted bits improved slightly over the `R3_Q8B` GPTQ baseline
- the allocator concentrated on `self_attn.v_proj` and `self_attn.k_proj`, with one `mlp.down_proj`

### Run G2R02_Q8B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-8B-Base`
- Method: `targeted_svd_rank`
- Base method: `gptq`
- Execution path: `Modal`
- GPU: `A100-80GB`
- Script: `./scripts/run_qwen3_8b_g2r02_gptq_modal.sh`
- Metrics dir: `results/modal/qwen3_8b_gptq_transfer/G2R02_Q8B`

Result:

- status: `completed`
- perplexity: `11.7962`
- memory: `6,109,349,763` bytes
- repair bytes: `5,373,952`

Interpretation:

- targeted rank was essentially flat relative to the `R3_Q8B` baseline
- targeted bits clearly beat targeted rank at the first matched `8B` GPTQ point

### Run G2B02RB_Q17B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_mixed_precision`
- Base method: `gptq`
- Action space: row-block bit upgrades, `256` rows per block
- Execution path: `Modal`
- GPU: `A10G`
- Manifest: `configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_richer_bits_manifest.json`
- Metrics dir: `results/modal/qwen3_1p7b_gptq_transfer/G2B02RB_Q17B`

Result:

- status: `completed`
- perplexity: `15.9060`
- memory: `1,370,337,577` bytes
- validation: finite logits and finite loss

Interpretation:

- the first richer-bits GPTQ pilot ran successfully on the cheaper `A10G` path
- it improved over the `R3_Q17B` GPTQ baseline
- but it did not beat the existing matrix-level bits point `G2B02_Q17B` (`15.8993`)
- so the first row-block design is not yet strong enough to justify spending `A100-80GB` time on an `8B` rerun

### Run G2B02RB128_Q17B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_mixed_precision`
- Base method: `gptq`
- Action space: row-block bit upgrades, `128` rows per block
- Execution path: `Modal`
- GPU: `A10G`
- Manifest: `configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_richer_bits_rowblock128_manifest.json`
- Metrics dir: `results/modal/qwen3_1p7b_gptq_transfer/G2B02RB128_Q17B`

Result:

- status: `completed`
- perplexity: `15.8970`
- memory: `1,370,468,649` bytes
- validation: finite logits and finite loss

Interpretation:

- the finer row-block design improved over the first `256`-row pilot
- it also beat the matrix-level bits point `G2B02_Q17B` (`15.8993`)
- this is the first richer-bits result strong enough to justify one `8B` validation run

### Run G2B02RB128_Q8B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-8B-Base`
- Method: `targeted_mixed_precision`
- Base method: `gptq`
- Action space: row-block bit upgrades, `128` rows per block
- Execution path: `Modal`
- GPU: `A100-80GB`
- Manifest: `configs/scaleup_qwen3_8b_gptq/qwen3_8b_gptq_richer_bits_rowblock128_manifest.json`
- Metrics dir: `results/modal/qwen3_8b_gptq_transfer/G2B02RB128_Q8B`

Result:

- status: `completed`
- perplexity: `11.7954`
- memory: `6,164,268,931` bytes
- validation: finite logits and finite loss

Interpretation:

- the richer-bits `128`-row design remains valid on `8B`
- it improves slightly over the `R3_Q8B` baseline
- but it does not beat the existing matrix-level bits point `G2B02_Q8B` (`11.7823`)
- so the richer-bits branch is now mixed across scale:
  - `1.7B`: richer bits improved over matrix-level bits
  - `8B`: matrix-level bits remained stronger

### Run G2R02F_Q17B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_svd_rank`
- Base method: `gptq`
- Action space: finer incremental rank ladder `2/4/6/8/12/16/24/32/48/64`
- Execution path: `Modal`
- GPU: `A10G`
- Manifest: `configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_fine_rank_manifest.json`
- Metrics dir: `results/modal/qwen3_1p7b_gptq_transfer/G2R02F_Q17B`

Result:

- status: `completed`
- perplexity: `15.9073`
- memory: `1,364,963,625` bytes
- repair bytes: `7,995,392`
- validation: finite logits and finite loss

Interpretation:

- the finer-rank ladder successfully spent more budget than the original `G2R02_Q17B`
- but it performed worse than the original matrix-level rank result (`15.8823`)
- so the next GPTQ branch should not be “more fine-rank ladder tuning”; it should be hybrid second-stage or a more structural rank action space

### Run G2B03RB128_Q17B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_mixed_precision`
- Base method: `gptq`
- Action space: row-block bit upgrades, `128` rows per block
- Budget: `+2.0%` of `R3_Q17B`
- Execution path: `Modal`
- GPU: `A10G`
- Manifest: `configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_richer_bits_rowblock128_2pct_manifest.json`
- Metrics dir: `results/modal/qwen3_1p7b_gptq_transfer/G2B03RB128_Q17B`

Result:

- status: `completed`
- perplexity: `15.9272`
- memory: `1,383,969,065` bytes
- validation: finite logits and finite loss

Interpretation:

- the richer-bits `+2.0%` follow-up was a valid run on `A10G`
- but it was clearly worse than the stronger `+1.0%` richer-bits point `G2B02RB128_Q17B` (`15.8970`)
- so, on `1.7B`, the second slice should not be assumed to go to more of the same row-block bit actions

### Run H2R02RB128_Q17B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `hybrid_second_stage`
- Base method: `gptq`
- Hybrid base: `G2B02RB128_Q17B`
- Extra budget slice: `+1.0%` of `R3_Q17B`
- Execution path: `Modal`
- GPU: `A10G`
- Manifest: `configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_hybrid_manifest.json`
- Metrics dir: `results/modal/qwen3_1p7b_gptq_transfer/H2R02RB128_Q17B`

Result:

- status: `completed`
- perplexity: `15.8989`
- memory: `1,374,466,345` bytes
- prior bit bytes: `13,500,416`
- second-stage rank bytes: `3,997,696`
- validation: finite logits and finite loss

Interpretation:

- the hybrid second-stage path is now implemented and working on Modal
- at `1.7B`, giving the next slice to rank after the best richer-bits point was much better than giving it to more richer bits (`G2B03RB128_Q17B`: `15.9272`)
- but hybrid still did not beat the earlier pure-rank `+1.0%` GPTQ point `G2R02_Q17B` (`15.8823`)
- so the strongest current takeaway is:
  - hybrid is useful as a second-stage correction relative to “more bits”
  - but it is not yet the best overall GPTQ strategy on `1.7B`

### Run G2R02RB128_Q17B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_svd_rank`
- Base method: `gptq`
- Action space: structural row-block rank repairs, `128` rows per block
- Execution path: `Modal`
- GPU: `A10G`
- Manifest: `configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_rowblock_rank_manifest.json`
- Metrics dir: `results/modal/qwen3_1p7b_gptq_transfer/G2R02RB128_Q17B`

Result:

- status: `completed`
- perplexity: `15.9034`
- memory: `1,370,523,945` bytes
- repair bytes: `13,555,712`
- validation: finite logits and finite loss

Interpretation:

- the first structural GPTQ rank pilot was valid and affordable on `A10G`
- but it underperformed the earlier matrix-level rank point `G2R02_Q17B` (`15.8823`)
- it also underperformed the stronger richer-bits point `G2B02RB128_Q17B` (`15.8970`)
- so the current evidence does not support spending more on small row-block rank variants

### Run H2R02_Q8B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-8B-Base`
- Method: `hybrid_second_stage`
- Base method: `gptq`
- Hybrid base: `G2B02_Q8B`
- Extra budget slice: `+1.0%` of `R3_Q8B`
- Execution path: `Modal`
- GPU: `A100-80GB`
- Manifest: `configs/scaleup_qwen3_8b_gptq/qwen3_8b_gptq_hybrid_manifest.json`
- Metrics dir: `results/modal/qwen3_8b_gptq_transfer/H2R02_Q8B`

Result:

- status: `completed`
- perplexity: `11.7895`
- memory: `6,155,487,107` bytes
- prior bit bytes: `46,137,344`
- second-stage rank bytes: `5,373,952`
- validation: finite logits and finite loss

Interpretation:

- `8B` hybrid second-stage repair improves over the `R3_Q8B` baseline and the rank-only point `G2R02_Q8B`
- but it still does not beat the bits-only point `G2B02_Q8B` (`11.7823`)
- so, under the current GPTQ action spaces, `8B` still prefers bits as the stronger policy

### Run G2B02CB128_Q17B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_mixed_precision`
- Base method: `gptq`
- Action space: structural column-block bit upgrades, `128` columns per block
- Execution path: `Modal`
- GPU: `A10G`
- Manifest: `configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_column_bits_manifest.json`
- Metrics dir: `results/modal/qwen3_1p7b_gptq_transfer/G2B02CB128_Q17B`

Result:

- status: `completed`
- perplexity: `15.9171`
- memory: `1,370,468,649` bytes
- validation: finite logits and finite loss

Interpretation:

- this was the first GPTQ bits pilot aligned directly to column-grouped quantization structure
- the run was valid and cheap on `A10G`
- but it underperformed the `R3_Q17B` baseline and every stronger `1.7B` bits variant already on disk
- so small column-block bit upgrades are not a promising next branch under the current scoring rule

### Run G2R02CB128_Q17B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_svd_rank`
- Base method: `gptq`
- Action space: structural column-block rank repairs, `128` columns per block
- Execution path: `Modal`
- GPU: `A10G`
- Manifest: `configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_column_rank_manifest.json`
- Metrics dir: `results/modal/qwen3_1p7b_gptq_transfer/G2R02CB128_Q17B`

Result:

- status: `completed`
- perplexity: `15.9004`
- memory: `1,370,521,897` bytes
- repair bytes: `13,553,664`
- validation: finite logits and finite loss

Interpretation:

- column-block rank is a better structural-rank fit than the earlier row-block rank pilot
- but it still underperformed the original matrix-level rank point `G2R02_Q17B` (`15.8823`)
- it also remained weaker than the stronger richer-bits point `G2B02RB128_Q17B` (`15.8970`)
- so the `1.7B` evidence now argues against spending more on small blockwise rank variants

### Run G2R02GRP_Q17B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_svd_rank`
- Base method: `gptq`
- Action space: matrix-level rank repairs with a family-aware grouped allocator
- Allocator: `greedy_family_round_robin` with two balanced family rounds
- Execution path: `Modal`
- GPU: `A10G`
- Manifest: `configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_grouped_rank_manifest.json`
- Metrics dir: `results/modal/qwen3_1p7b_gptq_transfer/G2R02GRP_Q17B`

Result:

- status: `completed`
- perplexity: `15.9224`
- memory: `1,360,965,929` bytes
- repair bytes: `3,997,696`
- validation: finite logits and finite loss

Interpretation:

- the grouped-rank allocator successfully spread the early budget across the main candidate families
- but the result was clearly worse than the original matrix-level rank point `G2R02_Q17B` (`15.8823`)
- this is a stronger negative result than the small blockwise pilots, because it changed the allocation policy itself rather than only the target shape
- so the next rank branch should not be another simple balancing heuristic on the same action set

### Run H2R02M_Q17B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `hybrid_second_stage`
- Base method: `gptq`
- Hybrid base: `G2B02_Q17B`
- Extra budget slice: `+1.0%` of `R3_Q17B`
- Execution path: `Modal`
- GPU: `A10G`
- Manifest: `configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_matrix_hybrid_manifest.json`
- Metrics dir: `results/modal/qwen3_1p7b_gptq_transfer/H2R02M_Q17B`

Result:

- status: `completed`
- perplexity: `15.8962`
- memory: `1,368,305,961` bytes
- prior bit bytes: `7,340,032`
- second-stage rank bytes: `3,997,696`
- validation: finite logits and finite loss

Interpretation:

- this run removed the earlier granularity confound by starting from the matrix-level bits policy
- hybrid improved over the first bits-only point `G2B02_Q17B`
- but it still did not beat `G2B03_Q17B` or `G2R02_Q17B`
- so the `1.7B` matrix-policy comparison is now clean:
  - rank-only best
  - bits-only second
  - hybrid third

### Run G2B03_Q8B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-8B-Base`
- Method: `targeted_mixed_precision`
- Base method: `gptq`
- Budget: `+2.0%` of `R3_Q8B`
- Execution path: `Modal`
- GPU: `A100-80GB`
- Manifest: `configs/scaleup_qwen3_8b_gptq/qwen3_8b_gptq_transfer_2pct_manifest.json`
- Metrics dir: `results/modal/qwen3_8b_gptq_transfer/G2B03_Q8B`

Result:

- status: `completed`
- perplexity: `11.8024`
- memory: `6,175,278,979` bytes
- validation: finite logits and finite loss

Interpretation:

- this was the missing equal-budget bits-only comparator for the `8B` hybrid result
- adding more bits beyond the first `+1.0%` slice made the result worse, not better
- so the best bits-only `8B` GPTQ point remains `G2B02_Q8B`

### Run G2R03_Q8B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-8B-Base`
- Method: `targeted_svd_rank`
- Base method: `gptq`
- Budget: `+2.0%` of `R3_Q8B`
- Execution path: `Modal`
- GPU: `A100-80GB`
- Manifest: `configs/scaleup_qwen3_8b_gptq/qwen3_8b_gptq_transfer_2pct_manifest.json`
- Metrics dir: `results/modal/qwen3_8b_gptq_transfer/G2R03_Q8B`

Result:

- status: `completed`
- perplexity: `11.7962`
- memory: `6,109,349,763` bytes
- repair bytes: `5,373,952`
- validation: finite logits and finite loss

Interpretation:

- this was the missing equal-budget rank-only comparator for the `8B` hybrid result
- it matched the earlier `G2R02_Q8B` result, which shows the current `8B` rank action space is saturated by `+1.0%`
- the final `8B` policy ordering is now:
  - bits-only best
  - hybrid second
  - rank-only third

### Run MB1_Q17B

- Date: 2026-03-14
- Model: `Qwen/Qwen3-1.7B-Base`
- Method: `targeted_mixed_precision`
- Base method: `gptq`
- Branch: `multi-bit bits-policy`
- Candidate bit widths: `5/6/8`
- Budget: `+1.0%` of `R3_Q17B`
- Execution path: `Modal`
- GPU: `A10G`
- Manifest: `configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_multibit_manifest.json`
- Metrics dir: `results/modal/qwen3_1p7b_gptq_multibit/MB1_Q17B`

Result:

- status: `completed`
- perplexity: `15.9097`
- memory: `1,366,667,561` bytes
- validation: finite logits and finite loss

Interpretation:

- this was the first and only justified run of the bounded multi-bit bits-policy branch
- the allocator selected a wide set of cheap `4->5` upgrades rather than `4->6` or `4->8`
- even so, it failed to beat the current best bits point `G2B02_Q17B` (`15.8993`)
- it also remained clearly below the current best rank point `G2R02_Q17B` (`15.8823`)
- this triggers the endgame stop rule:
  - do not run `MB2_Q17B`
  - do not continue to `8B`
  - treat the GPTQ multi-bit branch as tested and stopped at `1.7B`
