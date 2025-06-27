# 🧠 Task 4 - Logistic Regression (Breast Cancer Classification)

For this task, I worked on a binary classification problem using the Breast Cancer dataset. The goal was to predict whether a tumor is malignant or benign based on multiple features like radius, texture, symmetry, etc.

---

### 🔍 What I Did:

- Loaded the Breast Cancer dataset (from `sklearn` and saved as `breast_cancer.csv`)
- Standardized all numerical features using `StandardScaler`
- Used Logistic Regression for classification
- Evaluated the model using:
  - Confusion Matrix
  - Precision, Recall, F1-Score
  - ROC-AUC Curve
- Plotted the ROC Curve to visualize model performance

---

### 📁 Dataset Info:

- **File**: `breast_cancer.csv`
- **Rows**: 569
- **Target**: `target` (0 = malignant, 1 = benign)
- **Features**: 30 numerical features (like `mean radius`, `mean texture`, etc.)

---

### 🧪 Metrics:

- **Confusion Matrix** – Shows true vs predicted values
- **Classification Report** – Shows precision, recall, f1-score
- **ROC AUC Score** – Measures model’s ability to separate the two classes

---

### 🛠 Tools Used:

- `pandas`
- `matplotlib`, `seaborn`
- `scikit-learn`

---

### 💡 Learnings:

I understood how logistic regression can be used for binary classification, how evaluation metrics like precision/recall help in measuring performance, and the importance of standardizing features before training.
