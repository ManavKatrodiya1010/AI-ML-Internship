# 🚀 Titanic Dataset - Data Cleaning & Preprocessing (AI/ML Internship)

### 👋 About the Task

This was part of Task 1 for my AI & ML Internship. The goal was to take a real dataset (Titanic), clean it up, handle any missing values, encode categorical features, scale the numbers, and deal with outliers — all using Python.

---

### 🗃 Dataset Used

I used a version of the Titanic dataset with 714 records and 8 columns. It included fields like age, fare, passenger class, and survival status.

---

### 🛠️ Steps I Followed

1. **Loaded the CSV file** using Pandas.
2. **Explored the data** using `.info()`, `.describe()`, and `.head()`.
3. **Handled missing values** in the `age` column by filling them with the median.
4. **Encoded categorical columns** like `sex` using one-hot encoding.
5. **Standardized** the `age` and `fare` columns using `StandardScaler` from sklearn.
6. **Visualized outliers** in the `fare` column with a boxplot using Seaborn.
7. **Removed extreme outliers** (top 1%) from the `fare` column to keep the data clean.
8. **Saved the cleaned dataset** as a new CSV file for future use.

---

### 🧰 Libraries I Used

- `pandas` for working with data
- `numpy` for numerical operations
- `matplotlib` & `seaborn` for visualization
- `sklearn` for preprocessing (standardization)

---

### 📁 Files in This Repo

- `titanic.csv` → Original dataset
- `task.py` → Main Python script with preprocessing steps
- `titanic_cleaned.csv` → Final cleaned dataset
- `README.md` → This file (explaining everything)

---

### 📌 What I Learned

- How to detect and fix missing values
- How to encode categorical variables
- The difference between standardization and normalization
- How to find and remove outliers visually and statistically
- Why preprocessing is important before training ML models

---

### ✅ Final Output

A clean, scaled dataset saved as `titanic_cleaned.csv`, ready for training ML models!

s
