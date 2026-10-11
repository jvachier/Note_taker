# model2_cleanup

Seeds: [0, 1, 2] (mean ± std over seeds; the rule baseline is deterministic).

| Test set | Noise | System | Punct macro-F1 | Comma F1 | Period F1 | Question F1 | Casing acc | Capitalised F1 | UPPER F1 |
|---|---|---|---|---|---|---|---|---|---|
| wikitext | 0% | Rule baseline | 0.191 | 0.000 | 0.574 | 0.000 | 0.814 | 0.182 | 0.000 |
| wikitext | 0% | Model (model2_cleanup) | 0.671 ± 0.014 | 0.733 ± 0.001 | 0.896 ± 0.002 | 0.383 ± 0.044 | 0.976 ± 0.000 | 0.938 ± 0.001 | 0.944 ± 0.002 |
| wikitext | 5% | Rule baseline | 0.191 | 0.000 | 0.574 | 0.000 | 0.815 | 0.182 | 0.000 |
| wikitext | 5% | Model (model2_cleanup) | 0.630 ± 0.013 | 0.701 ± 0.000 | 0.865 ± 0.003 | 0.323 ± 0.039 | 0.969 ± 0.000 | 0.921 ± 0.001 | 0.926 ± 0.000 |
| wikitext | 10% | Rule baseline | 0.191 | 0.000 | 0.574 | 0.000 | 0.816 | 0.182 | 0.000 |
| wikitext | 10% | Model (model2_cleanup) | 0.613 ± 0.005 | 0.666 ± 0.002 | 0.840 ± 0.003 | 0.333 ± 0.013 | 0.961 ± 0.000 | 0.900 ± 0.001 | 0.906 ± 0.001 |
| wikitext | 20% | Rule baseline | 0.192 | 0.000 | 0.575 | 0.000 | 0.819 | 0.182 | 0.000 |
| wikitext | 20% | Model (model2_cleanup) | 0.557 ± 0.009 | 0.603 ± 0.002 | 0.784 ± 0.002 | 0.284 ± 0.031 | 0.945 ± 0.000 | 0.857 ± 0.001 | 0.858 ± 0.004 |
| dialogsum | 0% | Rule baseline | 0.093 | 0.000 | 0.280 | 0.000 | 0.876 | 0.514 | 0.000 |
| dialogsum | 0% | Model (model2_cleanup) | 0.789 ± 0.001 | 0.670 ± 0.002 | 0.850 ± 0.001 | 0.846 ± 0.002 | 0.965 ± 0.000 | 0.906 ± 0.001 | 0.808 ± 0.005 |
| dialogsum | 5% | Rule baseline | 0.093 | 0.000 | 0.280 | 0.000 | 0.875 | 0.503 | 0.000 |
| dialogsum | 5% | Model (model2_cleanup) | 0.756 ± 0.001 | 0.643 ± 0.001 | 0.823 ± 0.002 | 0.803 ± 0.001 | 0.959 ± 0.000 | 0.889 ± 0.001 | 0.789 ± 0.002 |
| dialogsum | 10% | Rule baseline | 0.094 | 0.000 | 0.281 | 0.000 | 0.873 | 0.494 | 0.000 |
| dialogsum | 10% | Model (model2_cleanup) | 0.724 ± 0.003 | 0.617 ± 0.005 | 0.796 ± 0.001 | 0.759 ± 0.002 | 0.952 ± 0.001 | 0.870 ± 0.002 | 0.788 ± 0.003 |
| dialogsum | 20% | Rule baseline | 0.094 | 0.000 | 0.282 | 0.000 | 0.870 | 0.474 | 0.000 |
| dialogsum | 20% | Model (model2_cleanup) | 0.662 ± 0.004 | 0.565 ± 0.003 | 0.740 ± 0.002 | 0.679 ± 0.006 | 0.937 ± 0.000 | 0.827 ± 0.001 | 0.729 ± 0.009 |

Validation punct macro-F1: 0.757 ± 0.002. Parameters: 7,290,119.
