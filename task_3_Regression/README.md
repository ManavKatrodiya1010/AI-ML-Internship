# 🏡 Task 3 - Linear Regression (House Price Prediction)

For this task, I worked on predicting house prices using linear regression. The dataset I used had around 100 rows with features like area, number of bedrooms, and bathrooms. The goal was to predict the price based on these inputs.

---

### 🔍 What I Did:

- Loaded the dataset (`house_prices_100.csv`) and did a quick check for missing values.
- Selected `area`, `bedrooms`, and `bathrooms` as input features.
- Split the data into training and test sets using `train_test_split`.
- Trained a Linear Regression model using `sklearn`.
- Evaluated the model using MAE, MSE, and R² score.
- Also plotted a graph to compare predicted vs actual prices for better understanding.

---

### 🧪 Evaluation Metrics:

These are the evaluation metrics I used to see how well the model is performing:

- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **R² Score** (Explains how well the model fits the data)

The values may differ each time due to randomness in train-test split, but overall the model gives decent results.

---

### 🗂 Files:

- `house_prices_100.csv` - The dataset with 100 entries
- `task.py` - The main Python code with model training and evaluation
- `README.md` - This file

---

### 🧰 Libraries Used:

- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

### 💡 Final Thoughts:

This task helped me understand how multiple linear regression works and how to interpret the results. I also got to see how the evaluation metrics actually reflect model performance. Simple and useful exercise!
