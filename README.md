# llm-decomposition

Repo for experiments on fixed-budget LLM compression, with the current focus on deciding whether extra memory should be spent on more bits, low-rank residual repair, or both.

Documentation is organized under [docs/README.md](docs/README.md):

- current and initial proposals
- consolidated execution roadmap
- phase results and decision memos
- detailed experiment journal
- archived source notes and feedback
- harness/config reference

Current stop-point:

- `RTN` results are complete through the current cross-scale regime map
- `GPTQ` matrix-level validation is now complete on `1.7B`, `3B`, and `8B`
- restart from [docs/roadmap/next_steps.md](docs/roadmap/next_steps.md)
- Kaggle notebook bring-up plan is in [docs/roadmap/kaggle_plan.md](docs/roadmap/kaggle_plan.md)
- Phase 3 roadmap is in [docs/roadmap/phase3_plan.md](docs/roadmap/phase3_plan.md)
- next implementation branch is [docs/roadmap/gptq_richer_action_space_plan.md](docs/roadmap/gptq_richer_action_space_plan.md)
