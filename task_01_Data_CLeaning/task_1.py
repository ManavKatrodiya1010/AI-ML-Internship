import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('titanic.csv')
print(df.info())
print(df.describe())
print(df.head())

print(df.isnull().sum())

df['age'] = df['age'].fillna(df['age'].median())

df = pd.get_dummies(df, columns=['sex'], drop_first=True)

scaler = StandardScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

sns.boxplot(x=df['fare'])
plt.title("Boxplot of Fare (before removing outliers)")
plt.show()

fare_threshold = df['fare'].quantile(0.99)
df = df[df['fare'] < fare_threshold]

df.to_csv("titanic_cleaned.csv", index=False)

print("\n Preprocessing complete. Cleaned dataset saved as 'titanic_cleaned.csv'")