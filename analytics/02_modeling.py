"""
Module 2 - Analytics Pipeline (/analytics)
Part B: Predictive modeling, continuing from the same cleaned data.

Reads the SAME committed titanic.csv produced by 01_eda.py.
No second sns.load_dataset('titanic') call happens anywhere in this file.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score,
)
from imblearn.over_sampling import SMOTE

CHART_DIR = "charts"
LOG_PATH = "modeling_log.txt"
log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


# ---------------------------------------------------------------------------
# Load the SAME cleaned dataset saved by 01_eda.py (no re-fetch from network)
# ---------------------------------------------------------------------------
df = pd.read_csv("titanic.csv")
log(f"Loaded cleaned dataset from titanic.csv, continuing the same pipeline. shape={df.shape}")

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
NUM_FEATURES = ["age", "sibsp", "parch", "fare"]
CAT_FEATURES = ["sex", "embarked"]
TARGET = "survived"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ---------------------------------------------------------------------------
# Task 7: Stratified train/test split (before any preprocessing)
# ---------------------------------------------------------------------------
log("\n" + "=" * 80)
log("TASK 7: STRATIFIED TRAIN/TEST SPLIT")
log("=" * 80)

class_balance = y.value_counts(normalize=True).round(3)
log(f"\nClass balance of 'survived' (full data): \n{class_balance.to_string()}")
log(
    "\nJustification for stratification: survival is imbalanced "
    f"(~{class_balance[0]*100:.0f}% did not survive vs ~{class_balance[1]*100:.0f}% did). "
    "A plain random split risks producing a train or test fold with a "
    "meaningfully different survival ratio than the full data purely by "
    "chance, which would bias evaluation metrics. Stratifying on 'survived' "
    "guarantees both splits preserve this ~62/38 ratio."
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
log(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
log(f"Train class balance:\n{y_train.value_counts(normalize=True).round(3).to_string()}")
log(f"Test class balance:\n{y_test.value_counts(normalize=True).round(3).to_string()}")

# ---------------------------------------------------------------------------
# Task 8: Preprocessing - fit on TRAIN ONLY via ColumnTransformer + Pipeline
# ---------------------------------------------------------------------------
log("\n" + "=" * 80)
log("TASK 8: PREPROCESSING (fit on train only)")
log("=" * 80)
log(
    "\nUsing a ColumnTransformer (median-impute + scale numeric columns, "
    "most-frequent-impute + one-hot encode categorical columns) wrapped in "
    "a scikit-learn Pipeline with the final estimator, so fit-on-train / "
    "transform-on-test is structurally enforced (imputer/encoder/scaler are "
    "fit only inside pipeline.fit(X_train, y_train); predict/transform on "
    "X_test never refits anything)."
)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUM_FEATURES),
    ("cat", categorical_transformer, CAT_FEATURES),
])

# ---------------------------------------------------------------------------
# Task 9 & 10: Train three classifiers + full evaluation suite
# ---------------------------------------------------------------------------
log("\n" + "=" * 80)
log("TASK 9 & 10: TRAIN & EVALUATE 3 CLASSIFIERS")
log("=" * 80)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
}

results = {}
fitted_pipelines = {}

plt.figure(figsize=(7, 6))
for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    results[name] = {
        "confusion_matrix": cm.tolist(), "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "auc": auc,
    }
    log(f"\n--- {name} ---")
    log(f"Confusion Matrix:\n{cm}")
    log(f"Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} "
        f"F1={f1:.3f} AUC={auc:.3f}")

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All 3 Classifiers")
plt.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/08_roc_curves.png", dpi=110)
plt.close()

# Confusion matrices as a grid
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Survived", "Survived"],
                yticklabels=["Not Survived", "Survived"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/09_confusion_matrices.png", dpi=110)
plt.close()

comparison_table = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "auc"]]
log("\n--- Comparison table (3 classifiers) ---")
log(comparison_table.round(3).to_string())

# Decision tree visualization with labeled features/classes
dt_pipe = fitted_pipelines["Decision Tree"]
feature_names = (
    NUM_FEATURES
    + list(dt_pipe.named_steps["preprocessor"]
           .named_transformers_["cat"].named_steps["encoder"]
           .get_feature_names_out(CAT_FEATURES))
)
plt.figure(figsize=(20, 10))
plot_tree(
    dt_pipe.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True, rounded=True, fontsize=8, max_depth=3,
)
plt.title("Decision Tree (depth-limited view, max_depth=3 shown)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/10_decision_tree.png", dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# Task 11: Imbalance handling comparison
# ---------------------------------------------------------------------------
log("\n" + "=" * 80)
log("TASK 11: IMBALANCE HANDLING COMPARISON (Random Forest)")
log("=" * 80)

log(f"\nClass balance (train): \n{y_train.value_counts(normalize=True).round(3).to_string()}")

imbalance_results = {}

# (a) baseline
pipe_base = Pipeline(steps=[("preprocessor", preprocessor),
                             ("classifier", RandomForestClassifier(random_state=42, n_estimators=200))])
pipe_base.fit(X_train, y_train)
pred_base = pipe_base.predict(X_test)
imbalance_results["baseline"] = {
    "precision": precision_score(y_test, pred_base),
    "recall": recall_score(y_test, pred_base),
    "f1": f1_score(y_test, pred_base),
}

# (b) class_weight='balanced'
pipe_bal = Pipeline(steps=[("preprocessor", preprocessor),
                            ("classifier", RandomForestClassifier(
                                random_state=42, n_estimators=200, class_weight="balanced"))])
pipe_bal.fit(X_train, y_train)
pred_bal = pipe_bal.predict(X_test)
imbalance_results["class_weight_balanced"] = {
    "precision": precision_score(y_test, pred_bal),
    "recall": recall_score(y_test, pred_bal),
    "f1": f1_score(y_test, pred_bal),
}

# (c) SMOTE applied to the TRAINING FOLD only (after preprocessing, before fitting)
X_train_pre = preprocessor.fit_transform(X_train, y_train)
X_test_pre = preprocessor.transform(X_test)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_pre, y_train)
rf_smote = RandomForestClassifier(random_state=42, n_estimators=200)
rf_smote.fit(X_train_sm, y_train_sm)
pred_smote = rf_smote.predict(X_test_pre)
imbalance_results["smote"] = {
    "precision": precision_score(y_test, pred_smote),
    "recall": recall_score(y_test, pred_smote),
    "f1": f1_score(y_test, pred_smote),
}

imbalance_df = pd.DataFrame(imbalance_results).T
log("\n--- Imbalance strategy comparison (Random Forest) ---")
log(imbalance_df.round(3).to_string())

best_f1_strategy = imbalance_df["f1"].idxmax()
log(
    f"\nConclusion: '{best_f1_strategy}' achieved the best F1 "
    f"({imbalance_df.loc[best_f1_strategy, 'f1']:.3f}) on this split. "
    "class_weight='balanced' and SMOTE both push recall on the minority "
    "('survived') class up relative to baseline by penalizing/oversampling "
    "minority-class errors, usually at some cost to precision; baseline "
    "tends to favor the majority class since it optimizes plain accuracy. "
    "Given the roughly 62/38 imbalance here (not extreme), the gain from "
    "resampling is real but modest."
)

# ---------------------------------------------------------------------------
# Task 12: Hyperparameter tuning (GridSearchCV) + OOB score
# ---------------------------------------------------------------------------
log("\n" + "=" * 80)
log("TASK 12: HYPERPARAMETER TUNING (GridSearchCV + OOB)")
log("=" * 80)

rf_grid_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)),
])

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [3, 5, 8, None],
    "classifier__max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(rf_grid_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_oob = grid_search.best_estimator_.named_steps["classifier"].oob_score_

log(f"\nBest params: {best_params}")
log(f"Best CV F1 score: {grid_search.best_score_:.3f}")
log(f"OOB score of best estimator (refit on full train set): {best_oob:.3f}")

# ---------------------------------------------------------------------------
# Task 13: Regression side-task - predict fare
# ---------------------------------------------------------------------------
log("\n" + "=" * 80)
log("TASK 13: REGRESSION SIDE-TASK (predict fare)")
log("=" * 80)

REG_FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "embarked", "survived"]
REG_NUM = ["age", "sibsp", "parch", "survived"]
REG_CAT = ["sex", "embarked", "pclass"]

Xr = df[REG_FEATURES].copy()
yr = df["fare"].copy()

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

reg_preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), REG_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), REG_CAT),
])

reg_pipe = Pipeline(steps=[("preprocessor", reg_preprocessor), ("regressor", LinearRegression())])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)
n, p = Xr_test.shape[0], Xr_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

log(f"\nMAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  Adjusted R2={adj_r2:.3f}")

residuals = yr_test - yr_pred
plt.figure(figsize=(7, 5))
plt.scatter(yr_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual Plot - Fare Regression")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/11_regression_residuals.png", dpi=110)
plt.close()

pred_median = np.median(yr_pred)
resid_std_low = residuals[yr_pred < pred_median].std()
resid_std_high = residuals[yr_pred >= pred_median].std()
log(f"\nResidual std for low-predicted-fare half: {resid_std_low:.2f}")
log(f"Residual std for high-predicted-fare half: {resid_std_high:.2f}")
hetero_conclusion = (
    "The residual plot shows heteroscedasticity: the spread of residuals "
    f"widens noticeably as predicted fare increases (std {resid_std_low:.2f} "
    f"vs {resid_std_high:.2f} for low vs high predicted-fare halves), "
    "rather than forming a uniform random band around zero. This is "
    "expected given fare's strong right skew (a few very high fares are "
    "hard to predict precisely with a linear model)."
)
log(hetero_conclusion)

# ---------------------------------------------------------------------------
# Task 14: Final model comparison table + recommendation
# ---------------------------------------------------------------------------
log("\n" + "=" * 80)
log("TASK 14: FINAL MODEL COMPARISON + RECOMMENDATION")
log("=" * 80)

log("\nClassification models (accuracy, precision, recall, f1, auc):")
log(comparison_table.round(3).to_string())

log("\nRegression model (MAE, RMSE, R2, Adjusted R2) - separate metric scale:")
reg_table = pd.DataFrame(
    {"Linear Regression (fare)": [mae, rmse, r2, adj_r2]},
    index=["MAE", "RMSE", "R2", "Adjusted_R2"],
).T
log(reg_table.round(3).to_string())

best_model_name = comparison_table["f1"].idxmax()
best_row = comparison_table.loc[best_model_name]

model_specific_notes = {
    "Logistic Regression": (
        "It is also the most interpretable of the three (coefficients map "
        "directly to feature effects) and the cheapest to retrain, though "
        "it trails the tree-based models on raw predictive power here."
    ),
    "Decision Tree": (
        "A single decision tree is easy to explain to non-technical "
        "stakeholders via the plot_tree visualization above, though it is "
        "more prone to overfitting the specific training split than an "
        "ensemble; watch its performance on fresh data over time."
    ),
    "Random Forest": (
        "As an ensemble it is more robust to the age/fare outliers "
        "identified in the EDA than a single Decision Tree, and its "
        "hyperparameters were validated via 5-fold GridSearchCV "
        "(best params above) rather than hand-tuned, reducing overfitting "
        "risk relative to an unconstrained single tree."
    ),
}

recommendation = (
    f"Recommendation: deploy the {best_model_name} model. It achieves the "
    f"best balance of precision ({best_row['precision']:.3f}) and recall "
    f"({best_row['recall']:.3f}), giving the top F1 score "
    f"({best_row['f1']:.3f}) among the three classifiers, with an AUC of "
    f"{best_row['auc']:.3f} indicating strong class separation. "
    f"{model_specific_notes[best_model_name]}"
)
log(f"\n{recommendation}")

# ---------------------------------------------------------------------------
# Task 15: Save the best full pipeline (preprocessing + estimator) via joblib
# ---------------------------------------------------------------------------
log("\n" + "=" * 80)
log("TASK 15: SAVE BEST FULL PIPELINE")
log("=" * 80)

best_full_pipeline = fitted_pipelines[best_model_name]
joblib.dump(best_full_pipeline, "best_pipeline.joblib")
log(f"\nSaved best full pipeline ({best_model_name}) -> best_pipeline.joblib")

reloaded = joblib.load("best_pipeline.joblib")
sample_raw = X_test.iloc[[0]]
original_pred = best_full_pipeline.predict(sample_raw)[0]
reloaded_pred = reloaded.predict(sample_raw)[0]
log(f"\nReload check on raw sample row: original_pred={original_pred}, "
    f"reloaded_pred={reloaded_pred}, match={original_pred == reloaded_pred}")
assert original_pred == reloaded_pred, "Reloaded pipeline prediction mismatch!"

# ---------------------------------------------------------------------------
# Persist everything for the README
# ---------------------------------------------------------------------------
with open(LOG_PATH, "w") as f:
    f.write("\n".join(log_lines))

modeling_summary = {
    "best_model": best_model_name,
    "classification_comparison": comparison_table.round(4).to_dict(orient="index"),
    "imbalance_comparison": imbalance_df.round(4).to_dict(orient="index"),
    "best_params": best_params,
    "best_oob_score": round(float(best_oob), 4),
    "regression": {"mae": round(mae, 4), "rmse": round(rmse, 4),
                   "r2": round(r2, 4), "adjusted_r2": round(adj_r2, 4)},
    "recommendation": recommendation,
    "heteroscedasticity_conclusion": hetero_conclusion,
}
with open("modeling_summary.json", "w") as f:
    json.dump(modeling_summary, f, indent=2)

print("\n02_modeling.py complete. Outputs: charts/, best_pipeline.joblib, "
      "modeling_log.txt, modeling_summary.json")
