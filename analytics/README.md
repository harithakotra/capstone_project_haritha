# Module 2 — Analytics Pipeline (`/analytics`)

Titanic dataset, loaded **once** via `sns.load_dataset('titanic')` and cached
to `titanic.csv`. `01_eda.py` does the load/clean/EDA; `02_modeling.py`
reads that same `titanic.csv` and continues straight into the modeling
pipeline — the raw dataset is never re-fetched.

Run order: `python3 01_eda.py` then `python3 02_modeling.py`.

Files:
- `01_eda.py` — Part A (profiling, cleaning, EDA, data story, standardization check)
- `02_modeling.py` — Part B (split, preprocessing, 3 classifiers, imbalance, tuning, regression, final pipeline)
- `titanic_raw.csv` — raw loaded frame, saved immediately after load
- `titanic.csv` — **the committed offline fallback**, cleaned dataset used by every later step (`pd.read_csv("titanic.csv")`)
- `charts/` — all chart images (`01_...png` – `11_...png`)
- `best_pipeline.joblib` — full fitted pipeline (preprocessing + best classifier), reloadable with `joblib.load`
- `eda_log.txt`, `modeling_log.txt`, `eda_summary.json`, `modeling_summary.json` — full console output / numeric summaries

---

## Part A — Profiling, cleaning, and the data story

### Task 1 — Profile
Raw shape: **891 rows × 15 columns**. Full `df.info()`/`df.describe()` output is in `eda_log.txt`.

Missing values (columns with any):

| Column | % missing |
|---|---|
| deck | 77.22% |
| age | 19.87% |
| embarked | 0.22% |
| embark_town | 0.22% |

### Task 2 — Missing-value handling (threshold rule)
- **`deck` (77.22%, >30%)** → **dropped the column**. At this missing rate, imputation would be fabricating data for over three-quarters of passengers; `deck` also duplicates signal already present in `pclass`/`fare` (cabin class is a strong proxy for deck location), so little information is lost.
- **`age` (19.87%, 5–30%)** → **imputed** with the median (28.0), the standard choice for a moderately-missing numeric column in this band.
- **`embarked` (0.22%, <5%)** → **dropped those rows**.
- **`embark_town` (0.22%, <5%)** → **dropped those rows** (same underlying missing records as `embarked`).

Cleaned shape after handling: **889 rows × 14 columns**, saved to `titanic.csv`.

### Task 3 — Univariate analysis (age, fare)
IQR outlier counts: **age → 65 outliers** (bounds [2.50, 54.50]); **fare → 114 outliers** (bounds [-26.76, 65.66]).

Fare: mean **32.10**, median **14.45**, mode **8.05**. Since mean > median > mode, **fare is right-skewed** — a small number of high-fare (first-class) passengers pull the mean well above the typical (median) fare.

Chart: `charts/01_univariate_age_fare.png`

### Task 4 — Bivariate analysis
Survival rate by **sex**: male 18.9%, female 74.0%.

Survival rate by **pclass**: 1st 62.6%, 2nd 47.3%, 3rd 24.2%.

Survival rate by **sex & pclass**:

| | 1st | 2nd | 3rd |
|---|---|---|---|
| Female | 96.7% | 92.1% | 50.0% |
| Male | 36.9% | 15.7% | 13.5% |

Correlation matrix (6 numeric columns only: `survived, pclass, age, sibsp, parch, fare`; `adult_male`/`alone` excluded as derived flags) — heatmap: `charts/02_correlation_heatmap.png`

**Two strongest correlations:**
1. **pclass ↔ fare: r = −0.548** — lower pclass number (better class) goes with higher fare, as expected since class is essentially fare-tier.
2. **sibsp ↔ parch: r = 0.415** — passengers traveling with siblings/spouses also tended to travel with parents/children, i.e. family groups cluster together rather than these being independent counts.

### Task 5 — Multivariate data story (4 charts)
1. **`charts/03_survival_by_class_sex.png`** (bar): Women survived at a much higher rate than men in every class, and first-class passengers of either sex out-survived third-class passengers of the same sex — sex and class compound rather than compete.
2. **`charts/04_age_by_survival.png`** (box): Survivors skew slightly younger with a wider lower-age spread, consistent with children being prioritized; the median gap is modest, so age alone is a weak predictor next to sex/class.
3. **`charts/05_fare_vs_age_scatter.png`** (scatter): Survivors cluster more densely at higher fares across all ages, reinforcing that ticket price (a proxy for class/deck) mattered more than age in isolation.
4. **`charts/06_pairplot.png`** (pairplot): The pclass-vs-fare panel shows the clearest separation between survivors and non-survivors; age contributes far less visible separation.

**Overall story:** survival was primarily driven by **sex and passenger class** (and the fare/deck access class implies), with age playing a secondary role mainly benefiting young children. Third-class men had the lowest survival odds of any group.

### Task 6 — Exploratory standardization check
Before: age mean 29.32 / std 12.98; fare mean 32.10 / std 49.70.
After z-scoring: age mean ≈ 0.00 / std ≈ 1.00; fare mean ≈ 0.00 / std ≈ 1.00 — confirmed. Chart: `charts/07_standardization_before_after.png`. This is an EDA-only sanity check; the modeling pipeline below fits its own scaler on the train split only.

---

## Part B — Predictive modeling

### Task 7 — Stratified split
Class balance: **62% not survived / 38% survived**. Stratifying on `survived` guarantees the train (61.7/38.3) and test (61.8/38.2) folds preserve this ratio rather than risking a skewed fold from a plain random split, which would bias evaluation metrics.

### Task 8 — Preprocessing
`ColumnTransformer` (median-impute + `StandardScaler` on `age, sibsp, parch, fare`; most-frequent-impute + `OneHotEncoder` on `sex, embarked`) wrapped in a `Pipeline` with the final estimator. All imputers/encoders/scalers are fit only via `pipeline.fit(X_train, y_train)`; test-set predictions never refit anything.

### Tasks 9–10 — Three classifiers, full evaluation

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.781 | 0.754 | 0.632 | 0.688 | 0.826 |
| Decision Tree | 0.815 | 0.787 | 0.706 | 0.744 | 0.821 |
| Random Forest | 0.803 | 0.780 | 0.676 | 0.724 | 0.823 |

Confusion matrices: `charts/09_confusion_matrices.png`. ROC curves: `charts/08_roc_curves.png`. Decision tree visualization (labeled features/classes, depth-limited to 3 for readability): `charts/10_decision_tree.png`.

### Task 11 — Imbalance handling comparison (Random Forest)

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.780 | 0.676 | 0.724 |
| class_weight='balanced' | 0.783 | 0.691 | **0.734** |
| SMOTE (train fold only) | 0.734 | 0.691 | 0.712 |

**Conclusion:** `class_weight='balanced'` gave the best F1 here. Both balancing strategies raise recall on the minority ("survived") class relative to baseline, usually at some precision cost; baseline favors the majority class since it optimizes plain accuracy. With a moderate ~62/38 imbalance (not extreme), the gain from resampling is real but modest, and simple class weighting edged out SMOTE on this split.

### Task 12 — Hyperparameter tuning (GridSearchCV, Random Forest)
Best params: `max_depth=5, max_features='sqrt', n_estimators=100`. Best CV F1: **0.744**. **OOB score of best estimator: 0.807**.

### Task 13 — Regression side-task (predict fare)
MAE **17.88**, RMSE **40.49**, R² **0.385**, Adjusted R² **0.360**. Residual plot: `charts/11_regression_residuals.png`.

**Heteroscedasticity conclusion:** the residual spread widens sharply as predicted fare increases (residual std ≈9.18 for the low-predicted-fare half vs ≈56.25 for the high-predicted-fare half) rather than forming a uniform random band around zero — this is **heteroscedastic**, driven by fare's strong right skew making the few very high fares hard for a linear model to predict precisely.

### Task 14 — Final comparison & recommendation

**Classification** (own scale — accuracy/precision/recall/F1/AUC, all 0–1):

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.781 | 0.754 | 0.632 | 0.688 | 0.826 |
| Decision Tree | 0.815 | 0.787 | 0.706 | 0.744 | 0.821 |
| Random Forest | 0.803 | 0.780 | 0.676 | 0.724 | 0.823 |

**Regression** (separate scale — dollar/variance-explained units, not comparable to the classification table above):

| Model | MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|---|
| Linear Regression (fare) | 17.877 | 40.493 | 0.385 | 0.360 |

**Recommendation:** deploy the **Decision Tree** model. It achieves the best balance of precision (0.787) and recall (0.706), giving the top F1 score (0.744) among the three classifiers, with an AUC of 0.821 indicating strong class separation. A single decision tree is also easy to explain to stakeholders via the `plot_tree` visualization, though it's more prone to overfitting the specific training split than an ensemble — its performance should be monitored on fresh data over time.

### Task 15 — Saved pipeline
Full fitted pipeline (preprocessing + Decision Tree estimator) saved via `joblib.dump(full_pipeline, "best_pipeline.joblib")`. Reload check: `joblib.load(...)` reproduces the identical prediction on a raw, unpreprocessed test row (`modeling_log.txt` confirms `match=True`).
