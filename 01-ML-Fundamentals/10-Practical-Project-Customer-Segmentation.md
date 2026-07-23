# Practical Project: Unsupervised Learning & Customer Segmentation

In our previous projects, we always had a "target" column (like `Churn` or `median_house_value`). This is Supervised Learning. 

In **Unsupervised Learning**, we have **no labels**. The algorithm must figure out the hidden structure of the data completely on its own. In this project, we will use **K-Means Clustering** to discover hidden customer segments based on their spending habits.

---

## Step 1: Generate Customer Data
To avoid any internet download errors, we will mathematically simulate a dataset of 500 customers with 3 features: Age, Annual Income, and Spending Score.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

# Simulate 500 customers grouped into 4 hidden behavioral clusters
X, _ = make_blobs(n_samples=500, n_features=3, centers=4, cluster_std=1.2, random_state=42)

# Convert to DataFrame and make numbers look realistic
df = pd.DataFrame(X, columns=['Age', 'Annual_Income', 'Spending_Score'])
df['Age'] = np.abs(df['Age'] * 10 + 35).astype(int)
df['Annual_Income'] = np.abs(df['Annual_Income'] * 15 + 60).astype(int)
df['Spending_Score'] = np.abs(df['Spending_Score'] * 10 + 50).astype(int)

# Notice there is NO 'Target' column here!
df.head()
```

---

## Step 2: Exploratory Data Analysis (EDA)
Let's see if we can spot the clusters with our own eyes using a 3D pairplot.

```python
# A pairplot shows every feature plotted against every other feature
sns.pairplot(df, diag_kind='kde', corner=True)
plt.suptitle('Customer Data Pairplot (Can you spot the clusters?)', y=1.02)
plt.show()
```

---

## Step 3: Scaling (Extremely Important!)
K-Means groups data based on physical distances (Euclidean distance). If Income is in the $60,000s and Age is in the 30s, Income will completely dominate the math. **We must scale the data.**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

print("Data scaled successfully!")
```

---

## Step 4: The Elbow Method (Finding 'K')
Since we don't have labels, how do we know how many clusters (K) to create? 
We use the **Elbow Method**. We test K from 1 to 10 and plot the WCSS (Within-Cluster Sum of Squares - basically, how tight the clusters are).

```python
from sklearn.cluster import KMeans

wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_) # Inertia is the WCSS

# Plot the Elbow
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.xticks(K_range)
plt.show()
```
*Look at the graph: the "elbow" is the point where adding more clusters doesn't improve the tightness significantly. (It should be around K=4).*

---

## Step 5: Final Clustering & PCA Visualization
Now we train the final K-Means model with $K=4$. 

Because we have 3 features (Age, Income, Spending), we can't easily plot it on a 2D screen. We will use **PCA (Principal Component Analysis)** to mathematically compress the 3 features into 2 features so we can visualize our customer segments!

```python
from sklearn.decomposition import PCA

# 1. Train final K-Means model
final_kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X_scaled)

# Add the predicted clusters back to our original dataframe
df['Customer_Segment'] = cluster_labels

# 2. Use PCA to compress 3D data down to 2D for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. Visualize the 4 customer segments
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='viridis', s=100, alpha=0.8)
plt.title('Customer Segments Visualized (via PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Segment')
plt.show()
```
