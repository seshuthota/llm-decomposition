| Quantizer | Model | Best PPL Policy | Best Mean Downstream Policy | Note |
| --- | --- | --- | --- | --- |
| RTN | Qwen3-0.6B | rank | n/a | smallest RTN point favors rank |
| RTN | Qwen3-1.7B | bits | rank (slight) | cross-quantizer flip vs GPTQ |
| RTN | SmolLM3-3B | bits | n/a | bits-favoring RTN midpoint |
| RTN | Qwen3-8B | bits | n/a | bits-favoring RTN large-scale point |
| GPTQ | Qwen3-1.7B | rank (single-seed) | bits | multiseed says within noise |
| GPTQ | SmolLM3-3B | mixed | bits | baseline best by PPL, bits best mean downstream |
| GPTQ | Qwen3-8B | bits | rank (near-tied) | bits stable across seeds; downstream nearly tied |
