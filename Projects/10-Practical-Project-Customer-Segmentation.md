# Practical Project: Unsupervised Learning & Customer Segmentation

In **Unsupervised Learning**, datasets lack predefined labels or target columns. The algorithm must mathematically discover the hidden structure of the data autonomously. This project utilizes **K-Means Clustering** to segment customers based strictly on behavioral features.

---

## Step 1: Generate Customer Data
To ensure a reliable testing environment, a simulated dataset of 500 customers is generated featuring three core metrics: Age, Annual Income, and Spending Score. 

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

# Simulate 500 customers grouped into 4 hidden behavioral clusters
X, _ = make_blobs(n_samples=500, n_features=3, centers=4, cluster_std=1.2, random_state=42)

# Convert to DataFrame
df = pd.DataFrame(X, columns=['Age', 'Annual_Income', 'Spending_Score'])
df['Age'] = np.abs(df['Age'] * 10 + 35).astype(int)
df['Annual_Income'] = np.abs(df['Annual_Income'] * 15 + 60).astype(int)
df['Spending_Score'] = np.abs(df['Spending_Score'] * 10 + 50).astype(int)

df.head()
```

---

## Step 2: Exploratory Data Analysis (EDA)
A 3D pairplot is utilized to visually inspect feature relationships prior to clustering.

```python
sns.pairplot(df, diag_kind='kde', corner=True)
plt.suptitle('Customer Data Pairplot', y=1.02)
plt.show()
```

---

## Step 3: Scaling (Mandatory for Distance-Based Algorithms)
K-Means groups data based on Euclidean distances. Because unscaled numerical features (e.g., a $60,000 income versus a 30-year-old age) will distort the distance calculations, data scaling via `StandardScaler` is an absolute necessity.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
```

---

## Step 4: The Elbow Method (Finding 'K')
Without supervised labels, the optimal number of clusters (`K`) is determined using the **Elbow Method**. The algorithm tests K from 1 to 10 and plots the Within-Cluster Sum of Squares (WCSS).

```python
from sklearn.cluster import KMeans

wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_) 

# Plot the Elbow
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.xticks(K_range)
plt.show()
```

---

## Step 5: Final Clustering & PCA Visualization
The final K-Means model is trained using the optimal $K=4$. 

Because 3-dimensional data cannot be easily plotted on a standard 2D axis, **Principal Component Analysis (PCA)** is employed to mathematically compress the 3 features down to 2 principal components for visualization purposes.

```python
from sklearn.decomposition import PCA

# 1. Train final K-Means model
final_kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X_scaled)

# Add the predicted clusters back to the original dataframe
df['Customer_Segment'] = cluster_labels

# 2. Use PCA to compress 3D data down to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. Visualize the customer segments
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='viridis', s=100, alpha=0.8)
plt.title('Customer Segments Visualized (via PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Segment')
plt.show()

# Output the mathematical centroid of each cluster
print(df.groupby('Customer_Segment').mean().round(1))
```

---

## Step 6: Results & Business Interpretation

The final `groupby` output generated the following mathematical segment averages:

```text
                  Age  Annual_Income  Spending_Score
Customer_Segment                                     
0                 76.3           78.9           143.0
1                 50.5          171.9            73.7
2                 53.7           42.4            20.4
3                 11.6          196.7            93.7
```

**Key Industry Insights & Marketing Actions:**
The primary value of Unsupervised Learning is converting mathematical groupings into actionable business profiles:
* **Segment 0 (The Elderly Big Spenders):** Retirees with moderate income but incredibly high spending scores. *Marketing Action: Target with premium, luxury items.*
* **Segment 1 (The Wealthy Savers):** Middle-aged demographics with massive incomes but moderate spending. *Marketing Action: Target with exclusive VIP incentivization programs.*
* **Segment 2 (The Budget-Conscious):** Middle-aged demographics with low income and very low spending. *Marketing Action: Target with discounts and budget-friendly products.*
* **Segment 3 (The Rich Kids):** Highly anomalous simulated cluster representing extreme youth with massive income and high spending (analogous in reality to children utilizing parental credit for micro-transactions). *Marketing Action: Target with digital goods and gaming ads.*
