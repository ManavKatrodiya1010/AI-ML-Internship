# 🌸 Task 6 - K-Nearest Neighbors (KNN) Classification (Iris Dataset)

For this task, I explored the Iris dataset and applied the K-Nearest Neighbors (KNN) algorithm to classify the species of flowers based on their features.

---

### 📋 What I Did:

- Loaded the Iris dataset and saved it to CSV
- Normalized the features using `StandardScaler` (important for distance-based algorithms)
- Trained a KNN classifier using `KNeighborsClassifier` from sklearn
- Experimented with values of K from 1 to 10
- Visualized accuracy for different values of K
- Evaluated the model using accuracy, confusion matrix, and classification report

---

### 📊 Dataset Info:

- **Dataset**: `iris.csv`
- **Samples**: 150
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Features**: Sepal length/width, Petal length/width

---

### 📈 Best K:

After trying multiple values, I found that **K = _N_** (fill with your best K) gave the highest accuracy.

---

### 🛠 Libraries Used:

- pandas
- matplotlib
- seaborn
- scikit-learn

---

### 🧠 Learnings:

- KNN is simple yet effective for classification
- Normalizing features is crucial for distance-based algorithms like KNN
- The value of K greatly impacts model performance
- Confusion matrix gives insight into misclassifications
