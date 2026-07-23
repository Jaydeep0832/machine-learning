# Evaluation Metrics & Class Imbalance

Relying solely on "Accuracy" is one of the most common mistakes made by beginners. In industry applications (like fraud detection or disease diagnosis), datasets are almost never balanced, making accuracy a misleading metric. This document covers the intuition, mathematics, and industry applications of robust evaluation metrics.

---

## 1. Precision

### Intuition
Precision answers the question: *"When the model claims it found something, how often is it right?"* It measures the reliability of a model's positive predictions.

### Formal Definition & Mathematics
Precision is the ratio of True Positive predictions to all Positive predictions made (both True Positives and False Positives).

$$
\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
$$

### When to Prioritize Precision
Prioritize Precision when the **cost of a False Positive is very high** (i.e., False Alarms are dangerous or expensive).
* **Spam Detection:** You want to ensure legitimate emails aren't flagged as spam.
* **Content Moderation:** You don't want to accidentally flag safe user posts and ban innocent users.
* **Financial Fraud:** You need high certainty before freezing a customer's credit card to avoid frustrating them.

---

## 2. Recall (Sensitivity / True Positive Rate)

### Intuition
Recall answers the question: *"Out of everything we actually wanted to find, how much did the model catch?"* It measures a model's ability to find all actual positive instances.

### Formal Definition & Mathematics
Recall is the ratio of correctly predicted positive cases to all true positive cases in the dataset.

$$
\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
$$

### When to Prioritize Recall
Prioritize Recall when the **cost of a False Negative is very high** (i.e., Missing something is catastrophic).
* **Medical Diagnosis:** Missing a life-threatening disease is fatal; you prefer false alarms over missed diagnoses.
* **Airport Security:** You must catch every prohibited item, even if it requires manually reviewing harmless items.
* **Self-Driving Cars:** The vehicle must detect every pedestrian in its path, even if it occasionally brakes for a harmless shadow.

---

## 3. The Precision-Recall Trade-off & F1-Score

### The Trade-off
Precision and recall exist in a constant tug-of-war. If you want a perfect recall score of 100%, you can simply predict "positive" for every single item. However, doing this floods your results with false positives, destroying your precision.

### The F1-Score (Harmonic Mean)
To evaluate both metrics together, data scientists use the F1-Score, which calculates the harmonic mean of precision and recall.

$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

* **Why Harmonic Mean?** A simple average is misleading. If Precision is $1.0$ and Recall is $0.0$, the simple average is $0.5$. However, a model with zero recall is utterly useless. The harmonic mean correctly punishes extreme imbalances, resulting in an F1-score of $0.0$ in this scenario.

### When to Use
* **Imbalanced Classes:** When detecting rare events, accuracy lies. F1-score exposes the flaw.
* **Equal Costs:** When False Positives and False Negatives carry roughly equal business consequences.
* **Search Engines:** You want relevant results on the first page (high precision) but you also don't want to miss major relevant documents (high recall).

---

## 4. ROC-AUC (Receiver Operating Characteristic)

### Intuition
While Precision, Recall, and F1 evaluate a model at a *specific* decision threshold (e.g., $0.5$), ROC-AUC tells you how well the model separates the two classes *globally* across all possible thresholds.

### Formal Definition
* **The ROC Curve:** Plots the True Positive Rate (Recall) on the Y-axis against the False Positive Rate ($\frac{FP}{FP+TN}$) on the X-axis.
* **AUC (Area Under the Curve):** Measures the entire 2D area underneath the ROC curve, giving a single score between 0 and 1.
  * `AUC = 1.0`: Perfect separation of classes.
  * `AUC = 0.5`: Random guessing (diagonal line).
  * `AUC = 0.0`: Perfectly inverted model.

### When to Use
* **Comparing Models:** The best metric for ranking different algorithms to see which one distinguishes classes best, regardless of the threshold.
* **Threshold Selection:** Visually pick the optimal threshold that balances TPR and FPR for your business needs.

> [!WARNING]
> **Common Mistake: ROC-AUC on Highly Imbalanced Data**
> ROC-AUC can look deceptively optimistic on heavily imbalanced datasets (e.g., fraud where only $0.01\%$ is positive). Because the FPR denominator includes the massive number of True Negatives, the FPR stays artificially low. In rare-event scenarios, always use a **Precision-Recall Curve (PR-AUC)** instead.

---

## 5. Handling Class Imbalance (Industry Perspective)

Before training, always check your class distribution.

### Python Implementation
```python
import pandas as pd

# Load your dataset
df = pd.read_csv('your_data.csv')

# Get percentages for each class
percentages = df['target_column'].value_counts(normalize=True) * 100
print("\n--- Class Percentages ---")
print(percentages)
```

### Interpreting Imbalance Severities
1. **Balanced (50:50 to 60:40):** Standard algorithms and metrics (like Accuracy and ROC-AUC) work perfectly.
2. **Mildly Imbalanced (60:40 to 70:30):** Generally safe to treat as balanced, but keep an eye on your minority class performance.
3. **Moderately Imbalanced (70:30 to 90:10):** Accuracy becomes a lying metric. You must start tracking F1-Score or Precision/Recall.
4. **Highly Imbalanced (< 10%):** Standard models will just guess the majority class. You must use specialized techniques like SMOTE (oversampling), class weighting, and evaluate strictly using PR-AUC (Precision-Recall AUC).
