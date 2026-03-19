| Model | HW | Batch | Policy | Tok/s | ms/token | Peak VRAM MB |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | A10G | 1 | baseline | 15.69 | 63.89 | 1488 |
| Qwen3-1.7B | A10G | 1 | bits | 18.48 | 54.12 | 4825 |
| Qwen3-1.7B | A10G | 1 | rank | 19.98 | 50.06 | 1634 |
| Qwen3-1.7B | A10G | 8 | baseline | 138.03 | 7.24 | 2031 |
| Qwen3-1.7B | A10G | 8 | bits | 140.39 | 7.13 | 5366 |
| Qwen3-1.7B | A10G | 8 | rank | 118.78 | 8.45 | 2178 |
| Qwen3-8B | A100 | 1 | baseline | 14.00 | 71.45 | 6178 |
| Qwen3-8B | A100 | 1 | bits | 13.78 | 72.59 | 6255 |
| Qwen3-8B | A100 | 1 | rank | 8.51 | 117.48 | 6445 |
| Qwen3-8B | A100 | 8 | baseline | 69.41 | 14.41 | 6958 |
| Qwen3-8B | A100 | 8 | bits | 112.05 | 8.93 | 22663 |
| Qwen3-8B | A100 | 8 | rank | 115.58 | 8.65 | 7230 |
