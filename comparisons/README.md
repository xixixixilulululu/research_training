# Cross-model comparison figures

These figures put ResNet50, VGG16 and DenseNet121 (the Task 2 fair-comparison models) side
by side. Each model has the same colour in every figure: ResNet50 is blue, VGG16 is orange
and DenseNet121 is green.

The figures only re-plot numbers that other tasks already produced. To regenerate them,
run this from the repo root. It needs no GPU.

```bash
python comparisons/make_comparisons.py
```

## Figures

| Figure | What it shows | Finding it supports | Data source |
|---|---|---|---|
| [`../roc_comparison.png`](../roc_comparison.png) | ROC curves of the three models (malignant = positive) | The AUCs are close together (0.91–0.93). DenseNet121 is strongest at very low false-positive rates, and ResNet50 is strongest through the middle of the curve. | `ROC_comparison.ipynb` |
| `cam_metrics_comparison.png` | Four Grad-CAM localisation metrics, one panel each, with dashed chance lines | ResNet50 puts the largest share of its CAM on the lesion and almost none on the frame border. DenseNet121's peak lands in the lesion most often, but its maps are about 3× wider. VGG16 has the most border activation. | `task5_cam/cam_summary.json` (`overall`) |
| `cam_correct_vs_wrong.png` | lesion_energy and border_energy, split into correct and wrong predictions | When a model is wrong, border_energy rises for all three models. lesion_energy falls only for VGG16. The dotted ticks show each group's own chance level, because the wrongly classified images have larger lesions. The wrong groups are small (n = 13 / 18 / 13). | `task5_cam/cam_summary.json` (`by_correct`) |
| `loss_curves_three_models.png` | Train and validation loss for each model, with the kept checkpoint marked | All three overfit after their best epoch: train loss keeps falling while val loss levels off. VGG16's val loss is the noisiest, and it is the only model that ran the full 20-epoch budget. | `task2_*/outputs/*_history.json` |
| `missed_case_probability_distribution.png` | P(malignant) for every missed malignant case (true = malignant, pred = benign) | Most misses are confident errors, not near misses. Only 2 / 1 / 1 of the 6 / 9 / 10 misses lie between 0.3 and 0.5. Lowering the threshold would recover few of them. | `task5_cam/<model>/cam_stats.csv` |
| `cam_model_agreement_matrix.png` | Mean per-image Pearson correlation between two models' CAMs | The models look at overlapping but clearly different regions (r = 0.40–0.50). The two 7×7-feature-map models, ResNet50 and DenseNet121, agree with each other most. | pairwise correlations printed by `task5_cam/GradCAM_comparison.ipynb` |

`roc_comparison.png` belongs to this group. It stays in the repo root only because it was
created there first and other documents link to that path.

For the full discussion behind these findings, see `../comparison.md` (Task 2 metrics and
ROC) and `../task5_cam/CAM_comparison.md` (Grad-CAM).
