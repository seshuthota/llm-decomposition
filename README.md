# llm-decomposition

Repo for experiments on fixed-budget LLM compression, with the current focus on deciding whether extra memory should be spent on more bits, low-rank residual repair, or both.

Documentation is organized under [docs/README.md](docs/README.md):

- current and initial proposals
- consolidated execution roadmap
- phase results and decision memos
- detailed experiment journal
- harness/config reference

Current stop-point:

- `RTN` results are complete through the `1.7B` transfer study
- `GPTQ` is paused at baseline bring-up
- restart from [docs/roadmap/next_steps.md](docs/roadmap/next_steps.md)
