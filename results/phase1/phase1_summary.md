# Phase 1 Summary

| Run | Method | Status | Memory (bytes) | Perplexity | Latency ms/token |
|-----|--------|--------|----------------|------------|------------------|
| R1 | full_precision | completed | 1192099840 | 16.844704810407233 | 0.6852036503778915 |
| R2 | rtn | completed | 307304448 | 30.516916135923783 | 0.6853402444656435 |
| R3 | gptq | pending | None | None | None |
| R4 | uniform_svd_repair | completed | 307550208 | 30.557722980280193 | 0.6853179166752205 |
| R5 | uniform_svd_repair | completed | 307795968 | 30.46538282428384 | 0.6852693627330594 |
| R6 | uniform_svd_repair | completed | 308287488 | 30.315506091916657 | 0.6852944902933895 |
| R7 | uniform_svd_repair | completed | 309270528 | 29.984764915103966 | 0.6853686295159577 |
| R8 | mixed_precision_budget_match | completed | 307304448 | 30.516916135923783 | 0.6855246653753161 |
| R9 | mixed_precision_budget_match | pending | None | None | None |
| R10 | mixed_precision_budget_match | completed | 307304448 | 30.516916135923783 | 0.685410345815119 |
| R11 | mixed_precision_budget_match | completed | 308877312 | 28.861768021353583 | 0.6849496552052786 |
