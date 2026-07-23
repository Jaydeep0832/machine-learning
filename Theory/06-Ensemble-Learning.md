# Ensemble Learning

Ensemble learning combines predictions from multiple individual models to create one final, stronger prediction. Instead of relying on a single "expert" model, it gathers a democratic vote from a crowd of models.

## 1. Why Use Ensemble Learning?

Ensemble methods dominate industry applications for tabular data because they systematically solve the two fundamental machine learning problems: **Bias (Underfitting)** and **Variance (Overfitting)**.

* **Higher Accuracy:** A diverse group of models usually outperforms any single algorithm.
* **Robustness:** Reduces the risk of a single model failing wildly on unseen data.
* **Overfitting Control:** Certain ensemble types excel at smoothing out noisy data, ensuring the model generalizes well to unseen datasets.

---

## 2. The 3 Main Types of Ensemble Learning

Most ensemble techniques fall into three primary categories: Bagging, Boosting, and Stacking.

### A. Bagging (Bootstrap Aggregating)

#### Intuition
Train many independent models on slightly different datasets and average their answers.

#### Mechanism
1. **Bootstrap Sampling:** Create multiple random subsets of the training data (sampled *with replacement*).
2. **Parallel Training:** Train an instance of the same algorithm (usually an unpruned Decision Tree) on each subset in parallel.
3. **Final Vote:** Take a majority vote for classification, or average the predictions for regression.

#### Advantages & Primary Goal
* **Goal:** Drastically reduces **Variance (Overfitting)**.
* **Famous Example:** **Random Forest** (an ensemble of independent Decision Trees).

### B. Boosting

#### Intuition
Train models sequentially, where each new model learns specifically from the mistakes of the previous one.

#### Mechanism
1. Train a base model on the data.
2. Evaluate its errors.
3. Train a *second* model, but place higher mathematical weight on the data points the first model misclassified.
4. Repeat sequentially. 
5. **Final Vote:** A weighted average where more accurate sequential models get a stronger say.

#### Advantages & Primary Goal
* **Goal:** Drastically reduces **Bias (Underfitting)**.
* **Famous Examples:** **AdaBoost**, **Gradient Boosting (GBM)**, and **XGBoost**. (Note: XGBoost is currently the industry standard for most tabular data problems).

### C. Stacking (Stacked Generalization)

#### Intuition
Train completely different algorithms and let a "boss" algorithm figure out who to trust.

#### Mechanism
1. **Base Learners:** Train highly diverse base models (e.g., an SVM, a Logistic Regression, and a Random Forest) on the same dataset.
2. **Predictions:** Take the predictions from all these models.
3. **Meta-Learner:** Instead of a simple vote, train a final "meta-model" (often a simple Logistic Regression) on those predictions to learn how to best combine them.

#### Advantages & Primary Goal
* **Goal:** Maximizes predictive power by blending entirely different algorithmic perspectives and feature representations. Highly popular in Kaggle competitions.

---

## 📊 Summary Comparison

| Technique | Execution | Solves For | Examples |
| :--- | :--- | :--- | :--- |
| **Bagging** | Parallel | High Variance (Overfitting) | Random Forest |
| **Boosting** | Sequential | High Bias (Underfitting) | XGBoost, AdaBoost, LightGBM |
| **Stacking** | Parallel + Meta-learner | Maximize overall accuracy | Custom Ensembles |
