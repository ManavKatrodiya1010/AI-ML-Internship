# Task 8: K-Means Clustering - Customer Segmentation

## 🎯 Objective

The goal of this task is to use unsupervised learning (K-Means) to segment mall customers based on their income, age, gender, and spending score.

## 🧾 Dataset

We used a synthetic version of the Mall Customer Segmentation dataset. It contains 200 customers with the following features:

- Gender (Male/Female)
- Age
- Annual Income (in thousands)
- Spending Score (1–100)

## 🧠 What I Did

1. **Data Loading & Preprocessing**

   - Loaded the CSV using Pandas.
   - Categorical column 'Gender' was encoded numerically.
   - Standardized the numerical features for better clustering.

2. **Elbow Method**

   - Applied the Elbow Method to find the optimal number of clusters using WCSS (inertia).
   - Based on the elbow plot, 5 clusters were chosen.

3. **Clustering**

   - Used `KMeans` from Scikit-learn to assign clusters to each customer.
   - Added cluster labels to the dataset.

4. **Visualization**

   - Plotted customer segments using Seaborn with color-coded clusters.

5. **Evaluation**
   - Calculated the **Silhouette Score** to evaluate how well the clustering performed.

## 🔍 Key Insights

- The data forms well-separated groups of customers with different spending habits.
- Cluster analysis helps businesses target marketing and tailor offers.

## 🛠 Tools Used

- Python, Pandas, Matplotlib, Seaborn, Scikit-learn

## ✅ Conclusion

K-Means is a simple yet effective algorithm for customer segmentation. The Elbow Method and Silhouette Score helped in selecting the best number of clusters.
