# Task 4: Disease Prediction from Medical Data
# Algorithms: SVM, Logistic Regression, Random Forest, XGBoost
# Datasets: Breast Cancer, Heart Disease, Diabetes (all three run in one go)
#
# Install first:  pip install pandas numpy scikit-learn xgboost matplotlib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             ConfusionMatrixDisplay)

# charts and the results table get saved in this folder
os.makedirs("outputs", exist_ok=True)


# ----------------------------------------------------------------
# STEP 1: A function that loads whichever dataset we ask for
# In every dataset, y = 1 means "has the disease", y = 0 means "healthy"
# ----------------------------------------------------------------
def load_data(dataset_name):

    if dataset_name == "breast_cancer":
        # comes built into scikit-learn
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        # scikit-learn uses 0 = cancer, so we flip it to match the others
        y = pd.Series(1 - data.target)

    elif dataset_name == "heart":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                   "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
        df = pd.read_csv(url, names=columns, na_values="?")
        # target is 0 (no disease) to 4 (severe), so make it just 0 or 1
        df["target"] = (df["target"] > 0).astype(int)
        # we do NOT drop the rows with missing values here, we fill them in
        # later instead, so we keep all 303 patients
        X = df.drop("target", axis=1)
        y = df["target"]

    elif dataset_name == "diabetes":
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                   "insulin", "bmi", "diabetes_pedigree", "age", "target"]
        df = pd.read_csv(url, names=columns)
        # In this file, a 0 in these columns really means "not measured"
        # (nobody has 0 blood pressure!), so mark them as missing
        for col in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
            df[col] = df[col].replace(0, np.nan)
        X = df.drop("target", axis=1)
        y = df["target"]

    return X, y


# ----------------------------------------------------------------
# STEP 2: Run the whole experiment for one dataset
# ----------------------------------------------------------------
def run_dataset(dataset_name):
    X, y = load_data(dataset_name)

    print("=" * 60)
    print("Dataset:", dataset_name)
    print("Rows and columns:", X.shape)
    print("Missing values:", int(X.isna().sum().sum()))
    print("Patients with disease:", int(y.sum()), "out of", len(y))
    print()

    # Split: 80% to learn from, 20% kept hidden for the final test.
    # stratify=y keeps the sick/healthy ratio the same in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fill missing values with the median (learned from training data only)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scale the features (SVM and Logistic Regression need this;
    # it does no harm to Random Forest and XGBoost, so we use it for all)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the four models
    models = {
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss"),
    }

    # Train each model and score it on the test patients
    rows = []
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        rows.append({
            "Dataset": dataset_name,
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        })

    results = pd.DataFrame(rows).round(3)
    print(results.drop(columns="Dataset").to_string(index=False))

    # Best model = highest F1 (balances catching sick patients and false alarms)
    best_name = results.sort_values("F1", ascending=False).iloc[0]["Model"]
    print()
    print("Best model:", best_name)
    print()
    print("Detailed report for", best_name)
    print(classification_report(y_test, predictions[best_name],
                                target_names=["No disease", "Disease"]))

    # Chart 1: confusion matrix of the best model
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions[best_name],
        display_labels=["No disease", "Disease"], cmap="Blues"
    )
    plt.title(dataset_name + " - " + best_name)
    plt.tight_layout()
    plt.savefig("outputs/" + dataset_name + "_confusion_matrix.png", dpi=150)
    plt.close()

    # Chart 2: which features mattered most (Random Forest)
    rf_model = models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    print("Top 5 important features:")
    print(importances.sort_values(ascending=False).head(5))
    print()

    importances.sort_values().tail(10).plot(kind="barh", color="steelblue")
    plt.title(dataset_name + " - top features (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("outputs/" + dataset_name + "_feature_importance.png", dpi=150)
    plt.close()

    return results


# ----------------------------------------------------------------
# STEP 3: Run all three datasets and save a combined table
# ----------------------------------------------------------------
all_results = []
for dataset_name in ["breast_cancer", "heart", "diabetes"]:
    all_results.append(run_dataset(dataset_name))

final = pd.concat(all_results, ignore_index=True)
final.to_csv("outputs/all_results.csv", index=False)

print("=" * 60)
print("SUMMARY - accuracy of every model on every dataset")
print(final.pivot(index="Model", columns="Dataset", values="Accuracy"))
print()
print("Charts and all_results.csv saved in the 'outputs' folder.")