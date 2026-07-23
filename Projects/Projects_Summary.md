# Practical Projects: Summary & Key Learnings

This document summarizes three practical industry-grade projects completed within this Machine Learning repository. Each project was implemented in Google Colab and focuses on a core area of Machine Learning: Classification, Regression, and Unsupervised Learning.

---

## 1. Telco Customer Churn Prediction (Classification)
**Objective:** Predict whether a customer will leave the telecom company based on their billing and demographic data.
**Algorithms Used:** Logistic Regression, Support Vector Machines (SVM), Random Forest.

### Key Learnings
* **The "Accuracy is a Liar" Principle:** The dataset was moderately imbalanced (~26% churn rate). A high accuracy (81%) is misleading because the model heavily favors guessing the majority class. Evaluation must rely on **Precision, Recall, and the F1-Score**.
* **Baselines Matter:** The simple Logistic Regression baseline model outperformed the complex SVM and Random Forest models out-of-the-box in terms of identifying churning customers (Recall of 0.57 vs 0.49).
* **Business Impact:** In industry, the cost of a False Negative (missing a churner) is high, making **Recall** the most critical metric to optimize.

### Industry Challenges Overcome
* Standard algorithms naturally struggle to identify the minority class. Future iterations require techniques like `SMOTE` (Synthetic Minority Oversampling) or applying `class_weights` to penalize the algorithm for missing churning customers.

---

## 2. California House Price Prediction (Regression)
**Objective:** Predict the continuous median house value in California districts based on features like income, location, and rooms.
**Algorithms Used:** Linear Regression, Random Forest Regressor.

### Key Learnings
* **Real-World Feature Engineering:** Implemented missing data handling via median imputation (for `total_bedrooms`) and One-Hot Encoding for categorical text variables (like `ocean_proximity`) to ensure mathematical processing compatibility.
* **Hyperparameter Tuning:** Utilized `GridSearchCV` to systematically test dozens of hyperparameters (like `n_estimators` and `max_depth`). The Optimized Random Forest achieved an **$R^2$ Score of 0.8182** and an **RMSE of 48,809**.
* **Regression Metrics:** Transitioned evaluation metrics from classification (Precision/Recall) to continuous error metrics (**RMSE** and **$R^2$**).

### Industry Challenges Overcome
* **Data Fetching Errors:** Encountered HTTP 403 authorization errors when utilizing the standard scikit-learn dataset loader in cloud environments. Overcame this by sourcing raw CSV data directly from reliable version-controlled repositories.
* **Computational Expense:** Demonstrated the intense computational expense of `GridSearchCV` on a 20,000-row dataset (training 54 models required over 5 minutes). `RandomizedSearchCV` serves as the standard industry alternative for speed.

---

## 3. Customer Segmentation (Unsupervised Learning)
**Objective:** Discover hidden groups of shoppers based on raw behavioral data (Age, Income, Spending Score) without any predefined labels or targets.
**Algorithms Used:** K-Means Clustering, Principal Component Analysis (PCA).

### Key Learnings
* **The Elbow Method:** Utilized the WCSS (Within-Cluster Sum of Squares) graph to mathematically determine that $K=4$ was the optimal number of clusters for the customer base.
* **Dimensionality Reduction:** Implemented **PCA** to compress 3-dimensional data down to 2 principal components, enabling visual plotting of the customer segments on a 2D scatter matrix.
* **Mandatory Scaling:** Verified that distance-based algorithms like K-Means fail completely if numerical features are not scaled using `StandardScaler`.

### Industry Challenges Overcome
* **Business Interpretation:** While the algorithm groups data mathematically, the core challenge is interpreting those groups for human stakeholders. By executing a `groupby().mean()` operation on the final clusters, the mathematical groups were successfully translated into actionable business segments (e.g., "The Wealthy Savers", "The Budget-Conscious").
