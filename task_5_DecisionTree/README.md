# ❤️ Task 5 - Decision Trees & Random Forest (Heart Disease Prediction)

In this task, I used a heart disease dataset to train two tree-based models: Decision Tree and Random Forest. The goal was to predict whether a person has heart disease based on medical attributes.

---

### 🔍 What I Did:

- Loaded and explored the dataset (`heart_disease.csv`)
- Used `DecisionTreeClassifier` with max depth to prevent overfitting
- Trained a `RandomForestClassifier` and compared it with the decision tree
- Evaluated both models using accuracy and classification report
- Used cross-validation for better accuracy estimation
- Plotted the decision tree and visualized feature importances from the random forest

---

### 📁 Dataset Info:

- **File**: `heart_disease.csv`
- **Rows**: 303
- **Features**: 13
- **Target**: `target` (0 = no disease, 1 = disease)

---

### 📊 Evaluation:

| Model         | Accuracy (Test) |
| ------------- | --------------- |
| Decision Tree | ~0.75–0.80      |
| Random Forest | ~0.80–0.85      |

Note: Accuracy may slightly vary due to random splits.

---

### 🛠 Libraries Used:

- `pandas`
- `matplotlib`, `seaborn`
- `scikit-learn`

---

### 💡 Learnings:

- Learned how decision trees work and how overfitting can be controlled using `max_depth`
- Saw how random forests improve performance by combining multiple trees (bagging)
- Understood how to interpret feature importance and tree structures
