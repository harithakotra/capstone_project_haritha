"""
Module 2 - Analytics Pipeline (/analytics)
Part A: Profiling, cleaning, and the data story

Loads the Titanic dataset ONCE via seaborn, profiles it, cleans it,
saves titanic.csv as the offline fallback, and produces the full EDA story.
02_modeling.py continues from the saved titanic.csv (no second load).
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

CHART_DIR = "charts"
LOG_PATH = "eda_log.txt"
log_lines = []


def log(msg=""):
    """Print and also collect for the written log / README source material."""
    print(msg)
    log_lines.append(str(msg))


# ---------------------------------------------------------------------------
# 1. Load once, profile, save offline fallback
# ---------------------------------------------------------------------------
log("=" * 80)
log("TASK 1: LOAD + PROFILE")
log("=" * 80)

df = sns.load_dataset("titanic")  # the ONE and ONLY raw load in this module

log("\n--- df.shape ---")
log(df.shape)

log("\n--- df.info() ---")
import io
buf = io.StringIO()
df.info(buf=buf)
log(buf.getvalue())

log("\n--- df.describe() ---")
log(df.describe(include="all").to_string())

# Save the raw loaded frame as the committed offline fallback immediately,
# per spec, before any cleaning is applied.
df.to_csv("titanic_raw.csv", index=False)

missing_pct = (df.isna().mean() * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
log("\n--- Missing value percentage (columns with any missing values) ---")
log(missing_pct.to_string())

# ---------------------------------------------------------------------------
# 2. Missing-value handling per the threshold rule
#    <5%  missing  -> drop those rows
#    5-30% missing -> impute
#    >30% missing  -> drop column OR encode "missing" as its own category
# ---------------------------------------------------------------------------
log("\n" + "=" * 80)
log("TASK 2: MISSING VALUE HANDLING")
log("=" * 80)

df_clean = df.copy()

for col, pct in missing_pct.items():
    log(f"\nColumn '{col}': {pct}% missing")
    if pct < 5:
        before = len(df_clean)
        df_clean = df_clean[df_clean[col].notna()]
        log(f"  -> {pct}% < 5%  => DROP ROWS. Rows {before} -> {len(df_clean)}")
    elif pct <= 30:
        if df_clean[col].dtype == object or str(df_clean[col].dtype) == "category":
            mode_val = df_clean[col].mode(dropna=True)[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            log(f"  -> 5%<= {pct}% <=30% => IMPUTE with mode ('{mode_val}')")
        else:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            log(f"  -> 5%<= {pct}% <=30% => IMPUTE with median ({median_val:.3f})")
    else:
        # >30% missing: imputation unreliable at this rate.
        # 'deck' is >70% missing in raw titanic -> too sparse to impute
        # meaningfully (cabin letter is essentially unrecoverable for most
        # passengers). We DROP the column rather than fabricate a "missing"
        # category, because deck has almost no signal once ~77% of rows are
        # unknown and it duplicates information already present in
        # pclass/fare (cabin class correlates with deck).
        log(f"  -> {pct}% > 30% => DROP COLUMN (imputation unreliable; "
            f"'{col}' duplicates signal already in pclass/fare)")
        df_clean = df_clean.drop(columns=[col])

log("\n--- Missing values remaining after cleaning ---")
log(df_clean.isna().sum().to_string())

# Save the CLEANED dataset as the one committed offline fallback used by
# every subsequent step (EDA charts below + 02_modeling.py).
df_clean.to_csv("titanic.csv", index=False)
log(f"\nSaved cleaned dataset -> titanic.csv  shape={df_clean.shape}")

# ---------------------------------------------------------------------------
# 3. Univariate analysis: age & fare
# ---------------------------------------------------------------------------
log("\n" + "=" * 80)
log("TASK 3: UNIVARIATE ANALYSIS (age, fare)")
log("=" * 80)


def iqr_outliers(series, name):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    log(f"{name}: Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} "
        f"bounds=[{lower:.2f}, {upper:.2f}] -> {len(outliers)} outliers")
    return len(outliers)


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df_clean["age"], kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Age - Histogram")
sns.boxplot(x=df_clean["age"], ax=axes[0, 1], color="steelblue")
axes[0, 1].set_title("Age - Box Plot")
sns.histplot(df_clean["fare"], kde=True, ax=axes[1, 0], color="darkorange")
axes[1, 0].set_title("Fare - Histogram")
sns.boxplot(x=df_clean["fare"], ax=axes[1, 1], color="darkorange")
axes[1, 1].set_title("Fare - Box Plot")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_univariate_age_fare.png", dpi=110)
plt.close()

age_outliers = iqr_outliers(df_clean["age"], "age")
fare_outliers = iqr_outliers(df_clean["fare"], "fare")

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]
log(f"\nfare: mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
if fare_mean > fare_median > fare_mode:
    skew_conclusion = (
        f"fare is RIGHT-SKEWED: mean ({fare_mean:.2f}) > median ({fare_median:.2f}) "
        f"> mode ({fare_mode:.2f}). A small number of high-fare (first-class) "
        f"passengers pull the mean up above the median."
    )
else:
    skew_conclusion = "See mean/median/mode values above for the ordering observed."
log(skew_conclusion)

# ---------------------------------------------------------------------------
# 4. Bivariate analysis
# ---------------------------------------------------------------------------
log("\n" + "=" * 80)
log("TASK 4: BIVARIATE ANALYSIS")
log("=" * 80)

# (a) survival rate by sex
log("\n-- Survival rate by sex --")
for s in df_clean["sex"].unique():
    mask = df_clean["sex"] == s
    rate = df_clean.loc[mask, "survived"].mean()
    log(f"  sex={s}: n={mask.sum()}, survival_rate={rate:.3f}")

# (b) survival rate by pclass
log("\n-- Survival rate by pclass --")
for p in sorted(df_clean["pclass"].unique()):
    mask = df_clean["pclass"] == p
    rate = df_clean.loc[mask, "survived"].mean()
    log(f"  pclass={p}: n={mask.sum()}, survival_rate={rate:.3f}")

# (c) survival rate by sex AND pclass (boolean masking with & )
log("\n-- Survival rate by sex & pclass --")
for s in df_clean["sex"].unique():
    for p in sorted(df_clean["pclass"].unique()):
        mask = (df_clean["sex"] == s) & (df_clean["pclass"] == p)
        rate = df_clean.loc[mask, "survived"].mean()
        log(f"  sex={s} & pclass={p}: n={mask.sum()}, survival_rate={rate:.3f}")

# Correlation matrix restricted to exactly these 6 numeric columns
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df_clean[corr_cols].corr()
log("\n-- Correlation matrix (6 numeric columns, adult_male/alone excluded) --")
log(corr.round(3).to_string())

# Find the two strongest off-diagonal absolute correlations
corr_pairs = []
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        corr_pairs.append((corr_cols[i], corr_cols[j], corr.iloc[i, j]))
corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
top2 = corr_pairs[:2]
log("\nTop 2 strongest |correlation| pairs:")
for a, b, v in top2:
    log(f"  {a} <-> {b}: r={v:.3f}")

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Heatmap (6 numeric columns)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_correlation_heatmap.png", dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 5. Multivariate "data story" (>= 4 distinct charts)
# ---------------------------------------------------------------------------
log("\n" + "=" * 80)
log("TASK 5: MULTIVARIATE DATA STORY (4+ charts)")
log("=" * 80)

# Chart 1: survival rate by sex & pclass (bar)
plt.figure(figsize=(7, 5))
sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex", errorbar=None)
plt.title("Chart 1: Survival Rate by Class and Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_survival_by_class_sex.png", dpi=110)
plt.close()
log(
    "\nChart 1 (bar - survival by class & sex): Women survived at a much "
    "higher rate than men in every class, and first-class passengers of "
    "either sex out-survived third-class passengers of the same sex. This "
    "shows sex and class acted as compounding, not competing, factors in "
    "who got to a lifeboat."
)

# Chart 2: age distribution by survival (box)
plt.figure(figsize=(7, 5))
sns.boxplot(data=df_clean, x="survived", y="age")
plt.title("Chart 2: Age Distribution by Survival")
plt.xlabel("Survived (0=No, 1=Yes)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_age_by_survival.png", dpi=110)
plt.close()
log(
    "\nChart 2 (box - age by survival): Survivors skew slightly younger and "
    "show a wider lower-age spread than non-survivors, consistent with "
    "children being prioritized during evacuation. The median age gap "
    "between the two groups is modest, so age alone is a weak predictor "
    "compared to sex or class."
)

# Chart 3: fare vs age scatter, colored by survival
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df_clean, x="age", y="fare", hue="survived", alpha=0.6, palette="Set1"
)
plt.title("Chart 3: Fare vs Age, colored by Survival")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_fare_vs_age_scatter.png", dpi=110)
plt.close()
log(
    "\nChart 3 (scatter - fare vs age by survival): Survivors (orange) "
    "cluster more densely at higher fares across all ages, while most "
    "high-fare, low-survival points are absent. This visually reinforces "
    "that ticket price (a proxy for class and deck location) mattered more "
    "than age in isolation."
)

# Chart 4: pair plot across key numeric features
pair_df = df_clean[["survived", "pclass", "age", "fare"]].copy()
pair_df["survived"] = pair_df["survived"].map({0: "No", 1: "Yes"})
g = sns.pairplot(pair_df, hue="survived", diag_kind="hist", palette="Set1")
g.fig.suptitle("Chart 4: Pairplot of pclass, age, fare by Survival", y=1.02)
g.savefig(f"{CHART_DIR}/06_pairplot.png", dpi=110)
plt.close("all")
log(
    "\nChart 4 (pairplot - pclass/age/fare by survival): The pclass-vs-fare "
    "panel shows the clearest separation between survivors and "
    "non-survivors, with survivors concentrated in lower pclass numbers "
    "(1st/2nd class) and higher fares. Age contributes far less visible "
    "separation, confirming class/fare as the dominant story."
)

log(
    "\nOverall data-story conclusion: survival on the Titanic was primarily "
    "driven by sex and passenger class (and the fare/deck access that class "
    "implies), with age playing a secondary role mainly benefiting young "
    "children. Third-class men had the lowest survival odds of any group."
)

# ---------------------------------------------------------------------------
# 6. Exploratory standardization check (age, fare) - EDA sanity check only
# ---------------------------------------------------------------------------
log("\n" + "=" * 80)
log("TASK 6: EXPLORATORY Z-SCORE STANDARDIZATION CHECK (age, fare)")
log("=" * 80)

before_stats = df_clean[["age", "fare"]].agg(["mean", "std"])
log("\nBEFORE standardization:")
log(before_stats.to_string())

scaler = StandardScaler()
scaled_vals = scaler.fit_transform(df_clean[["age", "fare"]])
scaled_df = pd.DataFrame(scaled_vals, columns=["age_z", "fare_z"])

after_stats = scaled_df.agg(["mean", "std"])
log("\nAFTER standardization (z = (x-mean)/std):")
log(after_stats.to_string())
log(
    "\nConfirmed: standardized age/fare have mean ~=0 and std ~=1 "
    "(this is an EDA-only sanity check; the actual modeling pipeline in "
    "02_modeling.py fits its own scaler on the TRAIN split only)."
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.kdeplot(df_clean["age"], ax=axes[0], label="age (raw)")
sns.kdeplot(scaled_df["age_z"], ax=axes[0], label="age (z-scored)")
axes[0].set_title("Age: before vs after standardization")
axes[0].legend()
sns.kdeplot(df_clean["fare"], ax=axes[1], label="fare (raw)")
sns.kdeplot(scaled_df["fare_z"], ax=axes[1], label="fare (z-scored)")
axes[1].set_title("Fare: before vs after standardization")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/07_standardization_before_after.png", dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# Persist the log for the README
# ---------------------------------------------------------------------------
with open(LOG_PATH, "w") as f:
    f.write("\n".join(log_lines))

# Also stash a few key numbers as a small json for 02_modeling.py / README use
import json

summary = {
    "raw_shape": list(df.shape),
    "clean_shape": list(df_clean.shape),
    "missing_pct": missing_pct.round(2).to_dict(),
    "age_outliers": age_outliers,
    "fare_outliers": fare_outliers,
    "fare_mean": round(fare_mean, 2),
    "fare_median": round(fare_median, 2),
    "fare_mode": round(fare_mode, 2),
    "top2_correlations": [[a, b, round(v, 3)] for a, b, v in top2],
}
with open("eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n01_eda.py complete. Outputs: titanic.csv, charts/, eda_log.txt, eda_summary.json")
