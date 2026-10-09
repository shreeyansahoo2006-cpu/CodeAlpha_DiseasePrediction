# CodeAlpha_DiseasePrediction

**Task 4: Disease Prediction from Medical Data** (CodeAlpha Machine Learning Internship)

This project predicts whether a patient has a disease from medical measurements such as age, blood tests and symptoms. It is a binary classification problem (1 = disease, 0 = no disease).

## Datasets
| Dataset | Patients | Features | Source |
|---|---|---|---|
| Heart Disease (Cleveland) | 303 | 13 | UCI ML Repository |
| Diabetes (Pima Indians) | 768 | 8 | UCI ML Repository |
| Breast Cancer Wisconsin (Diagnostic) | 569 | 30 | UCI ML Repository (via scikit-learn) |

## Algorithms
Support Vector Machine (SVM), Logistic Regression, Random Forest, XGBoost.

## Method
1. Load each dataset; turn the target into 0/1.
2. Diabetes: zeros in glucose, blood pressure, skin thickness, insulin and BMI are treated as missing values.
3. Split 80% train / 20% test (stratified, `random_state=42`).
4. Fill missing values with the median and scale features (fitted on training data only).
5. Train the four models and evaluate on the test set: accuracy, precision, recall, F1.
6. Save confusion matrices, feature-importance charts and `all_results.csv` in `outputs/`.

## Results (test accuracy)
| Model | Breast Cancer | Diabetes | Heart Disease |
|---|---|---|---|
| Logistic Regression | 0.965 | 0.708 | 0.869 |
| Random Forest | 0.974 | 0.773 | 0.885 |
| SVM | 0.974 | 0.740 | 0.852 |
| XGBoost | 0.965 | 0.766 | 0.852 |

Precision, recall and F1 for every model are in `outputs/all_results.csv`.

**Observations**
- Breast cancer is the easiest dataset (about 96-97% for all models).
- Diabetes is the hardest (71-77%); the data is noisy with limited measurements.
- Random Forest scored highest on heart disease and diabetes, and tied for highest on breast cancer.
- Test sets are small (e.g. 61 heart patients), so gaps of 1-2% between models are not strong evidence.
- Glucose and BMI were the most useful features for diabetes; worst area and worst concave points for breast cancer.

## How to run
```bash
pip install pandas numpy scikit-learn xgboost matplotlib
python3 "Disease Prediction.py"
```
Datasets for heart disease and diabetes download automatically. On macOS, XGBoost needs OpenMP (`brew install libomp`).

## Limitations
Small datasets from limited populations. This is a learning project and must not be used for real medical diagnosis.
