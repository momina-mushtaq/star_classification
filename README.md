# Stellar Classification with Logistic Regression

Classifying celestial objects as **Galaxy**, **Star**, or **Quasar (QSO)** from telescope measurements, using the Sloan Digital Sky Survey (SDSS17) dataset.

Built for an Artificial Intelligence lab course (Spring 2026).

## Results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
| --- | --- | --- | --- | --- |
| Logistic Regression (main model) | 92.90% | 0.93 | 0.91 | 0.92 |
| Decision Tree (extra experiment) | 96.46% | ~0.96 | ~0.96 | ~0.96 |

Evaluated on a held-out 20,000-sample test set (80/20 split, `random_state=42`).

**Per-class results, Logistic Regression**

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| GALAXY | 0.93 | 0.95 | 0.94 |
| QSO | 0.95 | 0.85 | 0.90 |
| STAR | 0.90 | 0.94 | 0.92 |

## Key takeaways

- **Redshift is the strongest signal.** KDE and violin plots show QSOs sit at much higher and more spread-out redshift than galaxies and stars.
- **A tree beat the linear model by about 3.5 points** (96.46% vs 92.90%), which suggests the class boundaries are non-linear. Logistic Regression can only draw linear boundaries.
- **QSOs are the hardest class for Logistic Regression** (recall 0.85), the main source of its errors.

## Dataset

[Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) (Kaggle): 100,000 observations, 18 columns.

| Class | Count | Share |
| --- | --- | --- |
| GALAXY | 59,445 | ~59% |
| STAR | 21,594 | ~22% |
| QSO | 18,961 | ~19% |

**Features used:** `u`, `g`, `r`, `i`, `z` (photometric filters), `redshift`, `alpha`, `delta` (sky coordinates).

**Dropped as identifiers / observation metadata:** `obj_ID`, `spec_obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `plate`, `MJD`, `fiber_ID`.

## Pipeline

1. **EDA:** shape, dtypes, `describe()`, unique counts, class balance
2. **Cleaning:** duplicate and null checks (none found)
3. **Visualization:** pie chart, redshift histogram, KDE, violin plot, pair plot of filters, correlation heatmap
4. **Preprocessing:** drop identifier/metadata columns, `LabelEncoder` on the target (GALAXY=0, QSO=1, STAR=2)
5. **Split and scale:** 80/20 `train_test_split`; `StandardScaler` fit on train only, applied to test (no data leakage)
6. **Modeling:** multinomial Logistic Regression (`solver='saga'`, `max_iter=1000`), plus a default Decision Tree for comparison
7. **Evaluation:** accuracy, macro precision/recall/F1, confusion matrices, one-vs-rest ROC curves with AUC

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn

## Repository structure

```
.
├── notebooks/
│   └── AI_Project_final.ipynb
├── data/
│   └── star_classification.csv
├── report/
│   └── AI_Project_Report.docx
└── README.md
```

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook notebooks/AI_Project_final.ipynb
```

## Limitations and next steps

- Logistic Regression hit `max_iter=1000` without fully converging with `saga`; trying `lbfgs` or more iterations is worth testing.
- The Decision Tree used default settings. Tuning depth, or trying Random Forest / Gradient Boosting, would likely improve results and reduce overfitting risk.
- QSO recall is the weak spot; class weighting or per-class threshold tuning could help.
- Cross-validation would give a more reliable estimate than a single split.

## Author

Momina Mushtaq - [GitHub](https://github.com/momina-mushtaq)
