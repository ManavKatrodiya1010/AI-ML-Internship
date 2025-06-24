import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('titanic.csv')

# 1. Summary statistics
print("🔍 Summary Statistics:")
print(df.describe())

# 2. Histograms for numeric features
numeric_cols = ['age', 'fare', 'sibsp', 'parch']
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Histogram of {col.capitalize()}')
    plt.xlabel(col.capitalize())
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# 3. Boxplots for numeric features
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col], color='orange')
    plt.title(f'Boxplot of {col.capitalize()}')
    plt.tight_layout()
    plt.show()

# 4. Pairplot
sns.pairplot(df[['age', 'fare', 'sibsp', 'parch', 'survived']], hue='survived')
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

# 5. Correlation matrix
plt.figure(figsize=(8, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# 🔎 Inference (optional — to be written in README)
print("\n✅ EDA Completed.")