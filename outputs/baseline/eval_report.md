# Baseline Rice Classification Evaluation Report

- **Model**: MobileNetV3-Small (8 Classes)
- **Manifest**: `splits/leak_free_split.csv` (Session-Grouped, Leakage-Free)
- **Training Device**: `cpu`
- **Total Training Time**: 1768.5s (29.47 min)
- **Best Epoch**: 3 (Val Macro-F1: 0.7766)

## Test Set Performance
- **Accuracy**: 89.74%
- **Macro-F1**: 0.7968
- **Test Loss**: 0.3294
- **Test Samples**: 4,953

### Per-Class Metrics

| Class | Category | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0_NOR` | Normal | 0.9805 | 0.9440 | 0.9619 | 3196 |
| `1_F&S` | Fusarium & Scab | 0.7489 | 0.7269 | 0.7377 | 238 |
| `2_SD` | Sprouted Damage | 0.8824 | 0.4393 | 0.5866 | 239 |
| `3_MY` | Moldy | 0.7581 | 0.7705 | 0.7642 | 244 |
| `4_AP` | Attacked by Pests | 0.5740 | 0.9132 | 0.7049 | 242 |
| `5_BN` | Broken | 0.8421 | 0.8703 | 0.8560 | 239 |
| `6_UN` | Unripe | 0.6959 | 0.9328 | 0.7971 | 238 |
| `7_IM` | Impurities | 0.9511 | 0.9811 | 0.9658 | 317 |

### Confusion Matrix
```text
True/Pred    0_NOR   1_F&S    2_SD    3_MY    4_AP    5_BN    6_UN    7_IM
--------------------------------------------------------------------------
0_NOR         3017       1       0       1     116      31      30       0
1_F&S            3     173       6      24      15       0      17       0
2_SD            31      45     105      19      27       2       7       3
3_MY            17       9       8     188       2       3      14       3
4_AP             7       2       0       3     221       0       9       0
5_BN             1       0       0       0       0     208      20      10
6_UN             1       1       0      12       2       0     222       0
7_IM             0       0       0       1       2       3       0     311
```
