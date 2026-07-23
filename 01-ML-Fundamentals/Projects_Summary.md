# Week 1 Practical Projects: Summary & Key Learnings

This document summarizes the three practical industry-grade projects completed during Week 1 (Machine Learning Fundamentals). Each project was implemented in Google Colab and focuses on a core area of Machine Learning: Classification, Regression, and Unsupervised Learning.

---

## 1. Telco Customer Churn Prediction (Classification)
**Objective:** Predict whether a customer will leave the telecom company based on their billing and demographic data.
**Algorithms Used:** Logistic Regression, Support Vector Machines (SVM), Random Forest.

### Key Learnings
* **The "Accuracy is a Liar" Principle:** The dataset was moderately imbalanced (~26% churn rate). We learned that a high accuracy (81%) is misleading because the model was simply guessing the majority class. We must rely on **Precision, Recall, and the F1-Score**.
* **Baselines Matter:** Surprisingly, the simple Logistic Regression baseline model outperformed the complex SVM and Random Forest models out-of-the-box in terms of finding the churning customers.
* **Business Impact:** In industry, the cost of a False Negative (missing a churner) is high, making **Recall** the most critical metric to optimize.

### Problems Faced
* Standard algorithms struggle to identify the minority class naturally. Future iterations require techniques like `SMOTE` (Synthetic Minority Oversampling) or applying `class_weights` to penalize the algorithm for missing churning customers.

---

## 2. California House Price Prediction (Regression)
**Objective:** Predict the continuous median house value in California districts based on features like income, location, and rooms.
**Algorithms Used:** Linear Regression, Random Forest Regressor.

### Key Learnings
* **Real-World Feature Engineering:** Learned how to handle missing data by imputing the median (for `total_bedrooms`) and how to One-Hot Encode categorical text variables (like `ocean_proximity`) so algorithms can process them mathematically.
* **Hyperparameter Tuning:** Implemented `GridSearchCV` to systematically test dozens of mathematical settings (like `n_estimators` and `max_depth`). The Optimized Random Forest achieved a massive **$R^2$ Score of 0.8182**.
* **Regression Metrics:** Transitioned away from classification metrics to using **RMSE** (Root Mean Squared Error) and $R^2$.

### Problems Faced
* **Data Fetching Errors:** Encountered a `HTTP 403: Forbidden` error when using the standard scikit-learn dataset loader. Overcame this by sourcing the raw CSV directly from a reliable GitHub repository.
* **Computational Expense:** Experienced firsthand how computationally expensive `GridSearchCV` is on a 20,000-row dataset, taking over 5 minutes to train 54 models. Learned that `RandomizedSearchCV` is the industry alternative for speed.

---

## 3. Mall Customer Segmentation (Unsupervised Learning)
**Objective:** Discover hidden groups of shoppers based on raw data (Age, Income, Spending Score) without any pre-existing labels or targets.
**Algorithms Used:** K-Means Clustering, Principal Component Analysis (PCA).

### Key Learnings
* **The Elbow Method:** Used the WCSS (Within-Cluster Sum of Squares) graph to mathematically determine that $K=4$ was the optimal number of clusters for the customer base.
* **Dimensionality Reduction:** Learned how to use **PCA** to compress 3-dimensional data (Age, Income, Spending) down to 2 principal components, making it possible to visualize the customer segments on a 2D scatter plot.
* **Mandatory Scaling:** Discovered that distance-based algorithms like K-Means will fail completely if features are not scaled using `StandardScaler` (e.g., a $60,000 income would dwarf a 30-year-old age).

### Problems Faced
* **Business Interpretation:** The algorithm grouped the data mathematically, but the challenge was interpreting those groups as a human. By using a `groupby().mean()` operation, we successfully profiled the clusters into actionable business segments (e.g., "The Elderly Big Spenders" vs "The Budget-Conscious").
