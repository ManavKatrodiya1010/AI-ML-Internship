import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset
df = pd.read_csv("mall_customers.csv")
print(df.head())

# Step 2: Encode categorical 'Gender' column
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male:1, Female:0

# Step 3: Feature selection
X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Elbow Method to find optimal k
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot WCSS vs. k
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.grid(True)
plt.show()

# Step 6: Fit final KMeans model (assume k=5)
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Step 7: Add cluster labels to DataFrame
df['Cluster'] = labels

# Step 8: Visualize clusters
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1')
plt.title("Customer Segments")
plt.show()

# Step 9: Silhouette Score
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.2f}")
