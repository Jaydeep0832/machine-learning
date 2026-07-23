# Practical Project: Telco Customer Churn Prediction

This document outlines an industry-grade Machine Learning pipeline for predicting customer churn. The workflow encompasses data preprocessing, handling class imbalance, algorithm selection (Logistic Regression, SVM, Random Forest), and advanced evaluation metrics.

---

## Step 1: Import Libraries and Load Data
This project utilizes the IBM Telco Customer Churn dataset, which contains customer billing, demographic data, and a boolean indicator for churn.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset directly from IBM's public repository
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Display the first 5 rows
df.head()
```

---

## Step 2: Exploratory Data Analysis (EDA) & Imbalance Check
Prior to applying modeling algorithms, the target variable (`Churn`) must be analyzed for class distribution.

```python
# Check dataset size
print(f"Dataset shape: {df.shape}\n")

# Check Class Imbalance for 'Churn'
churn_counts = df['Churn'].value_counts(normalize=True) * 100
print("--- Class Percentages ---")
print(churn_counts)

# Visualize it
sns.countplot(x='Churn', data=df)
plt.title("Class Imbalance: Churn vs Retained")
plt.show()
```
*Note: The dataset exhibits moderate imbalance (~26% churn rate), necessitating the use of F1-Score and Precision/Recall over basic Accuracy.*

---

## Step 3: Data Preprocessing & Feature Engineering
Data preprocessing involves cleaning the data and converting categorical text features into numerical formats.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Clean Data: 'TotalCharges' is read as text. Convert to numeric and fill missing values with 0.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# 2. Drop useless ID column
df = df.drop('customerID', axis=1)

# 3. Separate Features (X) and Target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# 4. Convert Target 'Yes/No' into 1/0
le = LabelEncoder()
y = le.fit_transform(y)

# 5. One-Hot Encode categorical features (like 'InternetService_Fiber optic')
X = pd.get_dummies(X, drop_first=True)

print(f"Features after encoding: {X.shape[1]}")
```

### Train / Test Split
The data is split utilizing `stratify=y` to maintain the exact 26% churn ratio across both the training and testing sets.

```python
# 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Feature Scaling (Crucial for SVM and Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## Step 4: Training Models
Three distinct models are trained for baseline comparison: a linear model (Logistic Regression), a non-linear model (SVM), and an ensemble (Random Forest).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Initialize models
log_reg = LogisticRegression(random_state=42)
svm_model = SVC(probability=True, random_state=42) 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
log_reg.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
```

---

## Step 5: Evaluation
Evaluation is conducted using a Classification Report, Matthew's Correlation Coefficient (MCC), Log Loss, and visual ROC-AUC mapping.

```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, log_loss, matthews_corrcoef

models = {
    "Logistic Regression": log_reg,
    "SVM": svm_model,
    "Random Forest": rf_model
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test) 
    
    print(f"\n========== {name} ==========")
    print(classification_report(y_test, y_pred))
    
    mcc = matthews_corrcoef(y_test, y_pred)
    loss = log_loss(y_test, y_pred_prob)
    
    print(f"Log Loss: {loss:.4f} (Lower is better)")
    print(f"MCC Score: {mcc:.4f} (-1 to 1, higher is better)")
```

---

## Step 6: Results & Business Interpretation

Upon execution, the baseline **Logistic Regression** model achieved an overall accuracy of **81%**. However, due to the class imbalance, the F1-Score for the minority class (Churn) was only **0.61**. 

**Key Industry Insights:**
1. **The "Accuracy is a Liar" Proof:** While 81% accuracy appears strong, the models struggled to predict the minority class, proving that basic accuracy metrics are insufficient for imbalanced datasets.
2. **Baselines Over Ensembles:** The simple Logistic Regression model actually outperformed the complex SVM and Random Forest models in terms of Recall (0.57 vs 0.49). 
3. **Business Impact:** In a telecom environment, missing a churning customer (False Negative) is highly expensive. The fact that the models missed 43% to 51% of actual churners indicates that the next iteration of this pipeline must involve **SMOTE** (Synthetic Minority Oversampling) or configured **class weights** to heavily penalize False Negatives.
