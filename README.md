# Breast Cancer Classification Using Logistic Regression

This project builds a binary classification model to detect breast cancer (Malignant or Benign) using logistic regression. The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains features computed from digitized images of fine needle aspirates of breast masses.

---

## 📁 Dataset Overview

- **Source**: `data.csv` (must be present in the working directory)
- **Target Variable**: `diagnosis` (M = Malignant, B = Benign → mapped to 1 and 0 respectively)
- **Number of Samples**: 569
- **Number of Features**: 30 numerical features

---

## 🔄 Preprocessing Steps

1. **Label Encoding**:
   - `diagnosis` column: M → 1, B → 0

2. **Column Cleanup**:
   - Removed columns: `id`, `Unnamed: 32` (if exists)

3. **Train-Test Split**:
   - Split: 75% train, 25% test
   - Stratified by diagnosis
   - `random_state=42` for reproducibility

4. **Feature Scaling**:
   - Used `StandardScaler` to normalize feature values

---

## 📊 Model Training

- **Model**: `LogisticRegression`
- **Library**: `scikit-learn`
- **Training Data**: Scaled features
- **Prediction**: Outputs both class and class probabilities

---

## 📈 Evaluation Metrics

- **Confusion Matrix**
- **Precision**: `0.9800`
- **Recall (Sensitivity)**: `0.9245`
- **ROC-AUC Score**: `0.9962`

### 🔍 ROC Curve
Shows the trade-off between the True Positive Rate and False Positive Rate.

---

## 📉 Threshold Tuning

Using the **default threshold (0.5)**:
- **Precision**: 0.9800
- **Recall**: 0.9245

Using a **lowered threshold (0.3)**:
- **New Precision**: 0.9808
- **New Recall**: 0.9623

> Lowering the threshold increases **recall** (detects more malignant cases), but slightly lowers **precision** (more false positives).

---

## 📐 Sigmoid Function in Logistic Regression

Logistic regression outputs probabilities by applying the **sigmoid function** to the linear combination of input features.

\[
P(y=1) = \frac{1}{1 + e^{-z}}
\]

Where `z` is the dot product of the features and weights (plus bias). This squashes the output into a [0,1] range, suitable for binary classification.

---

## 📊 Precision-Recall vs Threshold Plot

This plot helps understand how precision and recall vary with different classification thresholds. It’s particularly useful in imbalanced datasets where ROC curves may be misleading.

---

## ✅ Conclusion

- Logistic regression performs exceptionally well on this dataset.
- Tuning the decision threshold allows optimization for either **precision** (fewer false positives) or **recall** (fewer false negatives).
- Ideal for medical diagnosis problems where **recall is critical** (missing a malignant case is costly).

---

## 🛠️ Requirements

- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

---



