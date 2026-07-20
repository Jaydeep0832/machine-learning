# Logistic Regression & Classification

While the name contains "regression", Logistic Regression is actually a fundamental **classification** algorithm. It is used when the target variable is categorical (e.g., Spam or Not Spam, 0 or 1).

## 1. The Sigmoid Function

In classification, we want our predictions to be probabilities between 0 and 1. Linear regression outputs can be any real number ($-\infty$ to $\infty$). To map these outputs to a probability, we pass the linear hypothesis through a **Sigmoid (or Logistic) Function**.

**Sigmoid Function $g(z)$:**
$$ g(z) = \frac{1}{1 + e^{-z}} $$

**Hypothesis Function:**
$$ h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}} $$

This gives us the probability that $y=1$ given $x$, parameterized by $\theta$:
$$ h_\theta(x) = P(y=1 | x; \theta) $$

## 2. Decision Boundary

To make a final classification (0 or 1), we apply a threshold to the probability output:
- Predict $y = 1$ if $h_\theta(x) \geq 0.5$  (which happens when $\theta^T x \geq 0$)
- Predict $y = 0$ if $h_\theta(x) < 0.5$   (which happens when $\theta^T x < 0$)

The line or hyperplane where $\theta^T x = 0$ is called the **Decision Boundary**.

## 3. Cost Function (Log Loss / Cross-Entropy)

We cannot use Mean Squared Error for logistic regression because applying the non-linear sigmoid function makes the MSE cost function non-convex (meaning it has many local minima, making gradient descent fail).

Instead, we use the **Log Loss** (or Binary Cross-Entropy) cost function:

$$ Cost(h_\theta(x), y) = \begin{cases} -\log(h_\theta(x)) & \text{if } y = 1 \\ -\log(1 - h_\theta(x)) & \text{if } y = 0 \end{cases} $$

We can compress this into a single elegant equation:
$$ Cost(h_\theta(x), y) = -y \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x)) $$

**Full Cost Function $J(\theta)$:**
$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))] $$

## 4. Gradient Descent

Remarkably, the gradient descent update rule for logistic regression looks mathematically identical to the one for linear regression! The only difference is the definition of the hypothesis function $h_\theta(x)$.

**Update Rule:**
$$ \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} $$

## 5. Multiclass Classification (One-vs-All)

To classify data into more than two categories (e.g., $y \in \{0, 1, 2, ..., K\}$), we use the One-vs-All (One-vs-Rest) strategy:

1. Train $K$ separate binary logistic regression classifiers, one for each class $i$ to predict the probability that $y=i$.
   $$ h_\theta^{(i)}(x) = P(y=i | x; \theta) $$
2. For a new input $x$, make a prediction by picking the class $i$ that maximizes the probability.
   $$ \text{prediction} = \max_i ( h_\theta^{(i)}(x) ) $$
