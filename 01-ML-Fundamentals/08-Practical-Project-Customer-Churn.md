# Practical Project: Telco Customer Churn Prediction

This is a step-by-step guide designed for **Google Colab**. Follow these steps to implement everything you learned in Week 1 (Data Preprocessing, Class Imbalance, Logistic Regression, SVM, Random Forest, and Evaluation Metrics) into a single, industry-grade pipeline.

## 🚀 Setup: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Click **New Notebook**.
3. Create a new text cell at the top and paste the title: `# Telco Customer Churn Prediction`.

---

## Step 1: Import Libraries and Load Data
We will use a public dataset provided by IBM. It contains customer data (like monthly charges, tenure, and internet service) and whether they "Churned" (left the company) or not.

**Create a Code Cell in Colab and run:**
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
Before touching any ML algorithms, we must understand our data and check if the target variable (`Churn`) is balanced.

**Create a Code Cell and run:**
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
*Notice how only ~26% of customers churned. This is **moderately imbalanced**, meaning Accuracy will be a liar, and we must rely on the **F1-Score**!*

---

## Step 3: Data Preprocessing & Feature Engineering
Machine Learning models only understand numbers. We have to clean our data and convert text into numbers.

**Create a Code Cell and run:**
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
We split the data. Because of the imbalance, we use `stratify=y` so the 26% churn ratio is maintained in both the train and test sets.

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
Now we train three different models to see how the algorithms compare: Logistic Regression (Linear), SVM (Non-Linear), and Random Forest (Ensemble).

**Create a Code Cell and run:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Initialize models
log_reg = LogisticRegression(random_state=42)
svm_model = SVC(probability=True, random_state=42) # probability=True is needed for ROC-AUC
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
print("Training Logistic Regression...")
log_reg.fit(X_train, y_train)

print("Training Support Vector Machine...")
svm_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

print("Training Complete!")
```

---

## Step 5: Evaluation
Since the data is imbalanced, we will evaluate using a Classification Report (Precision, Recall, F1) and plot the ROC-AUC.

**Create a Code Cell and run:**
```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, log_loss, matthews_corrcoef

models = {
    "Logistic Regression": log_reg,
    "SVM": svm_model,
    "Random Forest": rf_model
}

# Evaluate Precision, Recall, and F1, plus extra metrics
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test) # Probabilities for both classes
    
    print(f"\n========== {name} ==========")
    print(classification_report(y_test, y_pred))
    
    # Calculate extra metrics
    mcc = matthews_corrcoef(y_test, y_pred)
    loss = log_loss(y_test, y_pred_prob)
    
    print(f"Log Loss: {loss:.4f} (Lower is better)")
    print(f"MCC Score: {mcc:.4f} (-1 to 1, higher is better)")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churn', 'Churn'], 
                yticklabels=['Not Churn', 'Churn'])
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Plot ROC Curves
plt.figure(figsize=(10, 6))

for name, model in models.items():
    # Get prediction probabilities for the positive class (Churn=1)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate False Positive Rate and True Positive Rate
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)
    
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], 'k--') # Diagonal random guess line
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR / Recall)')
plt.title('ROC-AUC Curve Comparison')
plt.legend()
plt.show()
```

## Step 6: Interpret Your Results
Look at the output in Colab and ask yourself:
1. Which model has the highest **F1-Score** for predicting `1` (Churn)?
2. If your business wants to catch *every* possible churning customer (even if it means accidentally giving discounts to happy customers), which metric should you look at? *(Hint: Look at Recall for class 1).*
3. How does the Logistic Regression baseline compare to the advanced Ensemble (Random Forest)?

**Congratulations! You have completed an industry-standard Machine Learning pipeline.**
