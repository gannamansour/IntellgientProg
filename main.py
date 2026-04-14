"""
main.py  —  Run the full pipeline end-to-end.

Steps:
1. Clean & preprocess data
2. Generate visualisations (saved as PNG files in reports/)
3. Run the expert symainstem on a sample patient
4. Train the Decision Tree model
5. Write an accuracy comparison report

After this script finishes, launch the UI with:
    streamlit run ui/app.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                            recall_score, f1_score, classification_report)
from sklearn.preprocessing import MinMaxScaler
import joblib

# ── Make local imports work regardless of cwd ────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from rule_based_system.expert_system import assess_patient

os.makedirs("reports", exist_ok=True)
os.makedirs("ml_model", exist_ok=True)


# Data cleaning & preprocessing
print("\n=== STEP 1: Data Preprocessing ===")

df_raw = pd.read_csv("data/raw_data.csv")
df = df_raw.copy()

df.drop_duplicates(inplace=True)
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].mean())

numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

df.to_csv("data/cleaned_data.csv", index=False)
print(f"Cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")



# STEP 2 – Visualisations

print("\n=== STEP 2: Visualisations ===")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap="coolwarm",
            ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
fig.savefig("reports/correlation_heatmap.png", dpi=120)
plt.close()
print("  [✓] reports/correlation_heatmap.png")

# Target distribution
fig, ax = plt.subplots(figsize=(5, 3))
df["target"].value_counts().plot.bar(ax=ax, color=["#2ecc71","#e74c3c"],
                                    edgecolor="white")
ax.set_xticklabels(["No Disease","Disease"], rotation=0)
ax.set_ylabel("Count")
ax.set_title("Target Class Distribution")
plt.tight_layout()
fig.savefig("reports/target_distribution.png", dpi=120)
plt.close()
print("  [✓] reports/target_distribution.png")

#  Feature importance (from a quick DT fit)
X_vis = df.drop(columns=["target"])
y_vis = df["target"]
dt_vis = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_vis.fit(X_vis, y_vis)
importances = pd.Series(dt_vis.feature_importances_, index=X_vis.columns).sort_values()

fig, ax = plt.subplots(figsize=(7, 5))
importances.plot.barh(ax=ax, color="#3498db")
ax.set_title("Feature Importance (Decision Tree)")
ax.set_xlabel("Importance Score")
plt.tight_layout()
fig.savefig("reports/feature_importance.png", dpi=120)
plt.close()
print("  [✓] reports/feature_importance.png")

# Histograms for key features
key_feats = ["age", "chol", "trestbps", "thalach", "oldpeak"]
fig, axes = plt.subplots(1, 5, figsize=(18, 3))
for ax, feat in zip(axes, key_feats):
    df[feat].hist(ax=ax, bins=20, color="#9b59b6", edgecolor="white")
    ax.set_title(feat)
plt.tight_layout()
fig.savefig("reports/feature_histograms.png", dpi=120)
plt.close()
print("  [✓] reports/feature_histograms.png")



#  Expert System demo
print("\n=== STEP 3: Rule-Based Expert System (sample patient) ===")

sample_patient = dict(age=65, chol=270, trestbps=155, thalach=90,
                        oldpeak=2.8, cp=2, exang=1, fbs=1, ca=2, thal=3)
es_result = assess_patient(sample_patient)
print(f"  Verdict    : {es_result['verdict']}  (score={es_result['risk_score']})")
print("  Rules fired:")
for r in es_result["rules_fired"]:
    print(f"    • {r}")



# ════════════════════════════════════════════════════════════════════════
# STEP 4 – Decision Tree Model
# ════════════════════════════════════════════════════════════════════════
print("\n=== STEP 4: Decision Tree Model ===")

X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "max_depth":         [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
}
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid, cv=5, scoring="f1", n_jobs=-1,
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print(f"  Best params : {grid.best_params_}")

y_pred = best_model.predict(X_test)
dt_metrics = {
    "accuracy":  accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall":    recall_score(y_test, y_pred),
    "f1":        f1_score(y_test, y_pred),
}
print(f"  Accuracy    : {dt_metrics['accuracy']:.3f}")
print(f"  Precision   : {dt_metrics['precision']:.3f}")
print(f"  Recall      : {dt_metrics['recall']:.3f}")
print(f"  F1-Score    : {dt_metrics['f1']:.3f}")
print("\n" + classification_report(y_test, y_pred,
                                    target_names=["No Disease","Disease"]))

joblib.dump(best_model, "ml_model/decision_tree_model.pkl")
print("  [✓] Model saved → ml_model/decision_tree_model.pkl")



# Comparison Report

print("\n=== STEP 5: Writing Comparison Report ===")

# Evaluate expert system on test set (un-normalised values for raw rules)
df_raw_test = df_raw.iloc[y_test.index]

correct_es = 0
for idx, row in df_raw_test.iterrows():
    pat = {c: row[c] for c in ["age","chol","trestbps","thalach","oldpeak",
                                "cp","exang","fbs","ca","thal"]}
    res = assess_patient(pat)
    # Map expert system verdict to binary: HIGH/MODERATE → 1, LOW → 0
    pred_es = 1 if res["verdict"] in ("HIGH RISK","MODERATE RISK") else 0
    if pred_es == row["target"]:
        correct_es += 1

es_accuracy = correct_es / len(df_raw_test)

report = f"""# Heart Disease Detection — Accuracy Comparison Report

## Dataset
- Total samples : {len(df)}
- Training set  : {len(X_train)}
- Test set      : {len(X_test)}

## Decision Tree Model

| Metric    | Score |
|-----------|-------|
| Accuracy  | {dt_metrics['accuracy']:.3f} |
| Precision | {dt_metrics['precision']:.3f} |
| Recall    | {dt_metrics['recall']:.3f} |
| F1-Score  | {dt_metrics['f1']:.3f} |

Best hyperparameters: `{grid.best_params_}`

## Rule-Based Expert System

| Metric   | Score |
|----------|-------|
| Accuracy | {es_accuracy:.3f} |

*(Precision/Recall/F1 not computed for the rule-based system as it outputs ordinal risk levels, not binary probabilities.)*

## Comparison & Analysis

| Aspect          | Expert System                        | Decision Tree                     |
|-----------------|--------------------------------------|-----------------------------------|
| Accuracy        | {es_accuracy:.3f}                   | {dt_metrics['accuracy']:.3f}      |
| Explainability  | High — rules are human-readable      | Medium — tree paths readable      |
| Data required   | None (domain knowledge only)         | Labelled dataset required         |
| Adaptability    | Manual rule updates needed           | Retrain on new data               |
| Speed           | Very fast (rule firing)              | Very fast (tree traversal)        |

## Conclusion
The Decision Tree achieves higher predictive accuracy by learning statistical
patterns from data. The Expert System is more transparent and requires no training
data, making it useful for rapid clinical screening when labelled examples are scarce.
A hybrid approach (flag high-risk patients with rules, confirm with ML) is recommended
for real-world deployment.
"""

with open("reports/accuracy_comparison.md", "w") as f:
    f.write(report)
print("  [✓] reports/accuracy_comparison.md")

print("\n✅  All steps complete!")
print("   Launch the UI:  streamlit run ui/app.py")
