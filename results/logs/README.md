# Run Logs

Manual run scripts write timestamped raw execution logs here.

These files are useful for:

- dependency failures
- runtime errors
- confirming exact start and finish time
- checking which script created a given result

They are not the best place to read experiment outcomes. For that, use:

- [../phase1/run_index.md](../phase1/run_index.md)
- [../phase1/phase1_summary.md](../phase1/phase1_summary.md)
- [../../docs/experiments/phase1_results.md](../../docs/experiments/phase1_results.md)

Primary entry points so far:

- `scripts/run_phase1_rl.sh`
- `scripts/run_r1_full_precision.sh`
- `scripts/run_r2_rtn_4bit.sh`
- `scripts/run_r3_gptq_4bit.sh`
- `scripts/run_r4_uniform_svd_rank4.sh`
- `scripts/run_r5_uniform_svd_rank8.sh`
- `scripts/run_r6_uniform_svd_rank16.sh`
- `scripts/run_r7_uniform_svd_rank32.sh`
- `scripts/run_r8_bits_match_r4.sh`
- `scripts/run_r9_bits_match_r5.sh`
- `scripts/run_r10_bits_match_r6.sh`
- `scripts/run_r11_bits_match_r7.sh`

Each invocation creates a new log file named like:

- `phase1_YYYYMMDD_HHMMSS.log`
- `install_gptq_deps_YYYYMMDD_HHMMSS.log`
