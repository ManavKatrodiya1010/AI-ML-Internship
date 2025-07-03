# Task 7: Support Vector Machine (SVM) Classifier

## 🔍 Objective

The goal of this task is to implement a Support Vector Machine (SVM) classifier using both linear and RBF kernels to classify breast cancer data into malignant or benign tumors.

## 📊 Dataset

The dataset used is the Breast Cancer Wisconsin dataset, which contains various features computed from images of breast masses.

## 📌 Steps Performed

1. Loaded the dataset and explored the features.
2. Split the data into training and test sets.
3. Applied feature scaling using `StandardScaler`.
4. Trained two SVM models:
   - One using a **linear kernel**.
   - One using an **RBF kernel**.
5. Evaluated both models using:
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-score)
6. Used **GridSearchCV** to tune hyperparameters (`C`, `gamma`) for the best performance.

## 📈 Results

- Both models performed very well on the test data.
- Grid Search helped find the optimal combination of hyperparameters.
- The RBF kernel slightly outperformed the linear one.

## ⚙️ Tools Used

- Python
- Scikit-learn
- Pandas, Matplotlib, Seaborn

## ✅ Conclusion

SVM is a powerful classification algorithm especially when the data is not linearly separable. Kernel tricks like RBF help extend the model’s capacity to fit complex patterns.
