# Practical Project: Advanced Regression & Hyperparameter Tuning

This project transitions from Classification (predicting discrete categories) to **Regression** (predicting continuous numerical values). The objective is to predict California House Prices based on features such as location, number of rooms, and median income. It also introduces the critical industry practice of **Hyperparameter Tuning** via `GridSearchCV`.

---

## Step 1: Import Libraries and Load Data
The dataset is loaded directly from a reliable raw CSV source. The dataset contains missing values and categorical data, requiring dedicated preprocessing steps.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

df.head()
```

---

## Step 2: Exploratory Data Analysis (EDA)
In regression tasks, the distribution of the target variable (`median_house_value`) must be analyzed. Heavy skewness can cause models to struggle with predicting extreme values.

```python
plt.figure(figsize=(8, 5))
sns.histplot(df['median_house_value'], bins=50, kde=True)
plt.title('Distribution of Median House Values')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.show()

# Look at correlations to see which feature affects price the most
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()
```

---

## Step 3: Data Cleaning, Encoding, & Scaling
Regression models rely heavily on spatial distances and gradients, making feature scaling mandatory. Additionally, missing values are imputed, and text columns are One-Hot Encoded.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Handle missing values in 'total_bedrooms' via median imputation
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# 2. One-Hot Encode the categorical 'ocean_proximity' column
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# 80% Training, 20% Testing (Stratification is not required for continuous targets)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Step 4: Baseline Models & Evaluation
A simple Linear Regression and a standard Random Forest Regressor are established as baselines. Evaluation is conducted using **RMSE** (Root Mean Squared Error) and **R² Score** (explained variance).

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Linear Regression Baseline
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
lin_preds = lin_reg.predict(X_test_scaled)

print("--- Linear Regression ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lin_preds)):.4f}")
print(f"R2 Score: {r2_score(y_test, lin_preds):.4f}\n")

# 2. Random Forest Baseline
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train_scaled, y_train)
rf_preds = rf_reg.predict(X_test_scaled)
```

---

## Step 5: Hyperparameter Tuning (GridSearchCV)
To optimize the Random Forest, **Grid Search** is utilized. This systematically tests various combinations of hyperparameters (using cross-validation internally) to discover the optimal model configuration.

```python
from sklearn.model_selection import GridSearchCV

# Define a "grid" of hyperparameter settings
param_grid = {
    'n_estimators': [50, 100, 200],      
    'max_depth': [None, 10, 20],         
    'min_samples_split': [2, 5]          
}

base_rf = RandomForestRegressor(random_state=42)

# Set up the Grid Search
grid_search = GridSearchCV(estimator=base_rf, param_grid=param_grid, 
                           cv=3, scoring='neg_root_mean_squared_error', 
                           verbose=2, n_jobs=-1)

grid_search.fit(X_train_scaled, y_train)

# Evaluate the winning model
best_model = grid_search.best_estimator_
best_preds = best_model.predict(X_test_scaled)
```

---

## Step 6: Results & Interpretation

The Grid Search identified the following optimal hyperparameter combination:
`{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}`

**Final Optimized Metrics:**
* **RMSE:** `48,809.9238`
* **R² Score:** `0.8182`

**Key Industry Insights:**
* **Variance Explained:** The final $R^2$ score of `0.8182` indicates that the Optimized Random Forest model can mathematically explain 81.8% of the variance in California house prices based purely on the provided features.
* **Computational Expense:** The Grid Search process required training 54 distinct Random Forest models, taking over 5 minutes. In industry applications with massive datasets, engineers often utilize `RandomizedSearchCV` to achieve comparable results at a fraction of the computational time.
