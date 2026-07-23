# Support Vector Machines & Decision Trees

This section covers two powerful, non-linear classification and regression algorithms: SVMs and Decision Trees.

## 1. Support Vector Machines (SVM)

SVM is a powerful classifier that works by finding the optimal hyperplane that maximizes the "margin" between different classes.

### The Large Margin Intuition
Unlike Logistic Regression which just wants to get the prediction on the correct side of the boundary, SVM wants the prediction to be as far away from the boundary as possible. The distance between the hyperplane and the closest data points (Support Vectors) is the margin.

### The Objective Function (Primal Form)
To maximize the margin, we must minimize the norm of the weight vector $w$. 

$$
\min_{w,b} \frac{1}{2} ||w||^2
$$

Subject to the constraint that all points are classified correctly:

$$
y^{(i)}(w^T x^{(i)} + b) \geq 1 \quad \text{for all } i
$$

### Soft Margin (Handling Outliers)
If data isn't perfectly separable, we introduce a regularization parameter $C$ and slack variables $\xi$:

$$
\min_{w,b,\xi} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{m} \xi^{(i)}
$$

Subject to: 
$$
y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi^{(i)}
$$

- **Large C**: Strict margin, lower bias, higher variance (prone to overfitting).
- **Small C**: Soft margin, higher bias, lower variance (ignores outliers).

### Kernels (Non-linear SVM)
When data is not linearly separable, SVM uses the "Kernel Trick" to project data into a higher-dimensional space where it *is* linearly separable, without actually computing the complex transformations.

**Gaussian (RBF) Kernel:**

$$
K(x, l) = \exp\left(-\frac{||x - l||^2}{2\sigma^2}\right)
$$

*(Where $\sigma^2$ controls how quickly the similarity falls off as distance increases).*

---

## 2. Decision Trees

Decision Trees build classification or regression models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed.

### Splitting Criteria

To decide which feature to split on at each node, the tree looks for the split that results in the purest child nodes.

**1. Gini Impurity (CART algorithm)**
A measure of how often a randomly chosen element from the set would be incorrectly labeled.

$$
\text{Gini} = 1 - \sum_{i=1}^{C} (p_i)^2
$$

*(Where $p_i$ is the probability of an item belonging to class $i$).*

**2. Entropy & Information Gain (ID3 algorithm)**
Entropy measures the amount of uncertainty or disorder in the data.

$$
H(S) = - \sum_{i=1}^{C} p_i \log_2(p_i)
$$

**Information Gain** is the decrease in entropy after a dataset is split on an attribute.

$$
IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)
$$

*(The tree chooses the attribute $A$ that maximizes Information Gain).*

### Random Forests
Decision trees are highly prone to overfitting. **Random Forests** solve this using an ensemble method:
1. Create many Decision Trees using bootstrapped samples of the training data.
2. At each split, only consider a random subset of features.
3. The final prediction is made by averaging (regression) or majority voting (classification) across all trees.
