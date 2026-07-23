# Practical Project: Advanced Regression & Hyperparameter Tuning

In this project, we move away from Classification (predicting categories) and tackle **Regression** (predicting exact numbers). We will predict California House Prices based on features like location, number of rooms, and median income.

We will also learn a critical industry skill: **Hyperparameter Tuning** using `GridSearchCV`.

---

## Step 1: Import Libraries and Load Data
We will use the California Housing dataset built directly into Scikit-Learn.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset directly from a reliable GitHub raw URL to avoid HTTP 403 errors
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

# Display the first 5 rows
df.head()
```

---

## Step 2: Exploratory Data Analysis (EDA)
In regression, we must check the distribution of our target variable (`MedHouseVal`). If it is heavily skewed, models struggle to predict extreme values.

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

## Step 3: Train / Test Split & Scaling
Regression models heavily rely on distances and gradients, so **Feature Scaling is mandatory**.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Handle missing values in 'total_bedrooms'
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# 2. One-Hot Encode categorical 'ocean_proximity'
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# 80% Training, 20% Testing (No stratify needed for regression!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Step 4: Baseline Models & Evaluation
We will train a simple Linear Regression and a standard Random Forest Regressor. We evaluate using **RMSE** (Root Mean Squared Error) and **R² Score** (how much variance the model explains; 1.0 is perfect).

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

print("--- Random Forest (Default) ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_preds)):.4f}")
print(f"R2 Score: {r2_score(y_test, rf_preds):.4f}")
```

---

## Step 5: Hyperparameter Tuning (GridSearchCV)
The default Random Forest is good, but we can make it better. An industry Data Scientist uses **Grid Search** to test hundreds of combinations of mathematical settings (hyperparameters) to find the absolute best model.

```python
from sklearn.model_selection import GridSearchCV

# We define a "grid" of settings we want to test
param_grid = {
    'n_estimators': [50, 100, 200],      # Number of trees
    'max_depth': [None, 10, 20],         # Maximum depth of each tree
    'min_samples_split': [2, 5]          # Minimum samples required to split an internal node
}

# Initialize a base model
base_rf = RandomForestRegressor(random_state=42)

# Set up the Grid Search (this runs Cross-Validation internally!)
grid_search = GridSearchCV(estimator=base_rf, param_grid=param_grid, 
                           cv=3, scoring='neg_root_mean_squared_error', 
                           verbose=2, n_jobs=-1)

# Fit it to the data (This will train 3 * 3 * 2 * 3(CV) = 54 different models! It takes a minute)
print("Starting Grid Search... This might take a minute.")
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest Parameters Found: {grid_search.best_params_}")

# Evaluate the winning model
best_model = grid_search.best_estimator_
best_preds = best_model.predict(X_test_scaled)

print("\n--- Optimized Random Forest ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, best_preds)):.4f}")
print(f"R2 Score: {r2_score(y_test, best_preds):.4f}")
```
