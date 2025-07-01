import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv('iris.csv')
print(df.head())

# 2. Split into features and target
X = df.drop('target', axis=1)
y = df['target']

# 3. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Try different K values
accuracies = []
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K={k}, Accuracy={acc:.2f}")

# 6. Plot accuracy vs K
plt.plot(range(1, 11), accuracies, marker='o')
plt.title("K vs Accuracy")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# 7. Final model with best K
best_k = accuracies.index(max(accuracies)) + 1
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
y_final_pred = final_model.predict(X_test)

# 8. Evaluation
print(f"Final Model Accuracy (K={best_k}):", accuracy_score(y_test, y_final_pred))
print(confusion_matrix(y_test, y_final_pred))
print(classification_report(y_test, y_final_pred))