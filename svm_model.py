# All imports
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Metrics computation function
def compute_metrics(y_true, y_pred, positive_label="P-MCI"):
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=positive_label)

    cm = confusion_matrix(y_true, y_pred, labels=["S-MCI", "P-MCI"])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)

    return acc, sens, spec


# Paths to datasets
training_dataset = "train_normalized_data.csv"
test_dataset = "test_normalized_data.csv"
output_folder = "svm_output"
os.makedirs(output_folder, exist_ok=True)

# Training data
df = pd.read_csv(training_dataset)
df = df[df["disease_state"].isin(["S-MCI", "P-MCI"])]


exclude_cols = ["samples", "age", "Sex", "disease_state"]
X = df.drop(columns=exclude_cols)
y = df["disease_state"]

print("Training class counts:\n", y.value_counts())

# Cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

coef_list = []
results = []  

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold   = X.iloc[val_idx]
    y_val_fold   = y.iloc[val_idx]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=1.0, max_iter=5000, random_state=42))
    ])

    pipeline.fit(X_train_fold, y_train_fold)

    # Collect coefficients
    coef_list.append(pipeline.named_steps["svm"].coef_[0])

    # Predictions for metrics
    y_pred_fold = pipeline.predict(X_val_fold)

    acc, sens, spec = compute_metrics(y_val_fold, y_pred_fold)

    results.append({
        "Fold": fold,
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec
    })

# Convert CV to dataframe
cv_results_df = pd.DataFrame(results)
cv_results_df.loc["Mean"] = cv_results_df.mean(numeric_only=True)

print("\nCross-Validation Performance:\n")
print(cv_results_df.round(3))

# Save CV metrics
cv_results_df.to_csv(f"{output_folder}/cv_metrics_SMCI_PMCI.csv", index=True)

# Biomarker identification
avg_coef = np.mean(np.abs(coef_list), axis=0)

gene_importance = pd.DataFrame({
    "gene": X.columns,
    "avg_abs_importance": avg_coef
}).sort_values(by="avg_abs_importance", ascending=False)

# Export top 30 genes
top_genes = gene_importance.head(30)
top_genes.to_csv(f"{output_folder}/LinearSVMtop_30_gene_biomarkers_SMCI_PMCI.csv", index=False)

print("\nTop 30 S-MCI vs P-MCI gene biomarkers exported successfully!")

# External cohort evaluation
test_df = pd.read_csv(test_dataset)

label_map = {
    "MCI": "S-MCI",
    "AD": "P-MCI"
}
test_df["disease_state"] = test_df["disease_state"].map(label_map)

print("\nExternal cohort label counts after mapping:")
print(test_df["disease_state"].value_counts())

# Filter valid samples
test_df = test_df[test_df["disease_state"].isin(["S-MCI", "P-MCI"])]
print("\nFiltered external cohort shape:", test_df.shape)

if test_df.shape[0] == 0:
    print("\nERROR: No S-MCI or P-MCI samples in the external test set.")
else:
    X_test = test_df.drop(columns=exclude_cols)
    y_test = test_df["disease_state"]
    X_test = X_test[X.columns]

    pipeline.fit(X, y)
    y_pred = pipeline.predict(X_test)

    # Classification report
    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True)
    ).transpose()

    print("\nExternal Cohort Classification Report:\n")
    print(report_df.round(3))
    report_df.to_csv(f"{output_folder}/classification_report_SMCI_PMCI.csv")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    cm_df = pd.DataFrame(cm, index=pipeline.classes_, columns=pipeline.classes_)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(f"{output_folder}/confusion_matrix_SMCI_PMCI.png", dpi=300, bbox_inches='tight')
    plt.show()

    # External cohort performance metrics
    acc_test, sens_test, spec_test = compute_metrics(y_test, y_pred)

    external_df = pd.DataFrame({
        "Dataset": ["External Cohort"],
        "Accuracy": [acc_test],
        "Sensitivity": [sens_test],
        "Specificity": [spec_test]
    })

    external_df.to_csv(f"{output_folder}/external_metrics_SMCI_PMCI.csv", index=False)
    print("\nExternal cohort metrics:\n", external_df.round(3))


# Summary table of metrics for training and external cohort 
summary_df = pd.DataFrame({
    "Dataset": ["Cross-Validation Mean", "External Cohort"],
    "Accuracy": [cv_results_df.loc["Mean", "Accuracy"], acc_test],
    "Sensitivity": [cv_results_df.loc["Mean", "Sensitivity"], sens_test],
    "Specificity": [cv_results_df.loc["Mean", "Specificity"], spec_test],
})

summary_df.to_csv(f"{output_folder}/metrics_summary_SMCI_PMCI.csv", index=False)

print("\nSummary Table:\n", summary_df.round(3))




