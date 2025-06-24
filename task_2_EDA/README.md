# 📊 Task 2: Exploratory Data Analysis (EDA)

This task focuses on understanding the dataset through visualizations and summary statistics. I used the Titanic dataset to explore distributions, correlations, and patterns.

---

### 📁 Dataset Used

- **Name**: titanic.csv
- **Columns**: age, fare, survived, pclass, sibsp, parch, sex, name

---

### 🔍 Steps I Performed

1. **Calculated summary statistics** using `.describe()`
2. **Plotted histograms** to see distributions of numeric features
3. **Created boxplots** to detect outliers
4. **Used pairplot** to see feature relationships by survival status
5. **Generated a correlation matrix** to find correlated features

---

### 📌 Observations

- Fare and age are slightly skewed.
- Sibling/Spouse (sibsp) and Parent/Child (parch) are mostly zero.
- Survivors tend to have slightly lower sibsp/parch counts.
- Fare and survival show some weak correlation.

---

### 🧰 Libraries Used

- `pandas` for data handling
- `matplotlib` and `seaborn` for plotting

---

### 📎 Files

- `task.py`: Python script with all EDA steps
- `titanic.csv`: Input dataset
- `README.md`: This file

---

### ✅ Outcome

This EDA helped me understand the relationships between features and survival, as well as find outliers and skewed distributions.
