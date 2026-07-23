# Clustering & Unsupervised Learning

Unsupervised learning algorithms draw inferences from datasets consisting of input data without labeled responses. The most common unsupervised learning task is clustering.

## 1. K-Means Clustering

K-Means aims to partition $m$ observations into $K$ clusters in which each observation belongs to the cluster with the nearest mean (cluster centroid).

### Algorithm Steps:
1. Randomly initialize $K$ cluster centroids $\mu_1, \mu_2, ..., \mu_K$.
2. **Cluster Assignment:** Assign each data point $x^{(i)}$ to the closest centroid.
   
$$
c^{(i)} := \arg \min_j || x^{(i)} - \mu_j ||^2
$$

3. **Move Centroids:** Update the position of each centroid to be the mean of all points assigned to it.
   
$$
\mu_j := \frac{1}{|S_j|} \sum_{i \in S_j} x^{(i)}
$$

   *(Where $S_j$ is the set of points assigned to cluster $j$)*.
4. Repeat steps 2 and 3 until convergence (centroids no longer move).

### Cost Function (Distortion)
K-Means intrinsically minimizes the squared distance between each point and its assigned centroid:

$$
J(c, \mu) = \frac{1}{m} \sum_{i=1}^{m} || x^{(i)} - \mu_{c^{(i)}} ||^2
$$

### Choosing K (The Elbow Method)
Plot the Cost $J$ as a function of the number of clusters $K$. The plot usually looks like an arm. The "elbow" of the arm is typically chosen as the optimal value for $K$.

---

## 2. Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique. It transforms a large set of variables into a smaller one that still contains most of the information in the large set.

### Goal of PCA
Reduce data from $n$-dimensions to $k$-dimensions by finding $k$ vectors onto which to project the data so as to minimize the projection error (the perpendicular distance from the data points to the projected surface).

### PCA Algorithm
1. **Data Preprocessing:** Mean normalization and feature scaling.
   
$$
x_j := \frac{x_j - \mu_j}{s_j}
$$

2. **Compute Covariance Matrix $\Sigma$:**
   
$$
\Sigma = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)})(x^{(i)})^T = \frac{1}{m} X^T X
$$

3. **Compute Eigenvectors (Singular Value Decomposition):**
   Using linear algebra libraries, compute SVD on $\Sigma$:
   
$$
[U, S, V] = \text{svd}(\Sigma)
$$

   The matrix $U$ contains the principal components (eigenvectors).
4. **Reduce Dimensionality:**
   Extract the first $k$ columns of $U$, resulting in $U_{\text{reduce}}$ (an $n \times k$ matrix).
   Compute the new reduced features $z$:
   
$$
z = U_{\text{reduce}}^T x
$$


### Variance Retained
To check how much information (variance) is retained when reducing to $k$ dimensions, we can use the diagonal matrix $S$ from the SVD:

$$
\text{Variance Retained} = \frac{\sum_{i=1}^{k} S_{ii}}{\sum_{i=1}^{n} S_{ii}}
$$

Typically, we choose $k$ such that 95% or 99% of the variance is retained.
