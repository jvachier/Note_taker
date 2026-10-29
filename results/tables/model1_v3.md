# Model 1 v3: meeting notes (test, gold labels)

Synthetic = `meeting-notes-extractive-v1` synthetic test; **AMI = real meetings, the headline**. Students: mean ± std over seeds. Teachers' rows use their cached test outputs (Notebook 1); thresholds are tuned on validation. HN FP = share of gold hard negatives kept (lower is better). Spans: owner / deadline exact-match F1 (synthetic test).

| System | Params | Synth imp F1 | Synth zone F1 | Synth HN FP ↓ | AMI imp F1 | AMI PR-AUC | AMI dec F1 | AMI act F1 | AMI R-1 | AMI R-2 | AMI R-L | AMI dec R-L | Spans F1 | CPU ms (100 / 500 sent.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Lead-k | – | 0.337 | – | 0.490 | 0.319 | – | – | – | 0.148 | 0.041 | 0.076 | – | – | – |
| Keyword rule | – | 0.136 | 0.247 | 0.000 | 0.029 | – | 0.045 | 0.024 | 0.128 | 0.020 | 0.076 | 0.082 | – | – |
| Qwen3-0.6B (teacher) | 600.0M | 0.403 | 0.194 | 0.585 | 0.318 | 0.210 | 0.030 | 0.000 | 0.143 | 0.041 | 0.073 | 0.066 | – | – |
| RoBERTa-Large (fine-tuned, ceiling) | 355.0M | 0.903 | – | 0.070 | 0.634 | 0.641 | – | – | 0.166 | 0.054 | 0.087 | – | – | – |
| DistilBERT (fine-tuned) | 66.0M | 0.870 | – | 0.187 | 0.580 | 0.555 | – | – | 0.149 | 0.050 | 0.080 | – | – | – |
| model1_v3_E1 | 20.9M | 0.878 ± 0.013 | 0.620 ± 0.039 | 0.142 ± 0.044 | 0.578 ± 0.006 | 0.550 ± 0.005 | 0.081 ± 0.014 | 0.016 ± 0.023 | 0.154 ± 0.003 | 0.050 ± 0.001 | 0.082 ± 0.001 | 0.098 ± 0.027 | 0.808 ± 0.069 | 405.6 / 2150.9 |
| model1_v3_E4 | 20.9M | 0.884 ± 0.006 | 0.609 ± 0.062 | 0.154 ± 0.029 | 0.567 ± 0.006 | 0.552 ± 0.014 | 0.071 ± 0.051 | 0.007 ± 0.009 | 0.161 ± 0.002 | 0.051 ± 0.001 | 0.085 ± 0.001 | 0.057 ± 0.040 | 0.688 ± 0.140 | 405.1 / 2158.3 |
