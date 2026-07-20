# Linear Regression & Gradient Descent

Linear regression is a foundational supervised learning algorithm used for predicting a continuous target variable based on one or more input features.

## 1. Hypothesis Function

The goal of linear regression is to fit a line (or hyperplane in higher dimensions) that best maps the inputs to the output.

For a single feature $x$, the hypothesis is:
$$ h_\theta(x) = \theta_0 + \theta_1 x $$

For multiple features $x_1, x_2, ..., x_n$:
$$ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n $$

In vectorized form, where $X$ is the feature matrix and $\theta$ is the parameter vector:
$$ h_\theta(X) = X \theta $$

## 2. Cost Function (Mean Squared Error)

To measure how well our line fits the data, we use a Cost Function $J(\theta)$. In linear regression, the most common is the Mean Squared Error (MSE), which measures the average squared difference between our predictions and the actual values $y$.

$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 $$
*Where $m$ is the number of training examples.*

## 3. Gradient Descent

To find the optimal parameters $\theta$ that minimize the cost function $J(\theta)$, we use an optimization algorithm called Gradient Descent. It works by taking steps in the opposite direction of the gradient of the function.

**Update Rule:**
Repeat until convergence:
$$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) $$
*Where $\alpha$ is the learning rate.*

**Calculating the Partial Derivative:**
$$ \frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} $$

So the final update rule becomes:
$$ \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} $$

## 4. Normal Equation (Analytical Solution)

For smaller datasets, we can solve for $\theta$ analytically without needing to iterate using Gradient Descent. This is done by setting the derivative of the cost function to zero and solving for $\theta$:

$$ \theta = (X^T X)^{-1} X^T y $$

*Note: This operation involves calculating the inverse of a matrix, which has a time complexity of $O(n^3)$. Therefore, Gradient Descent is preferred when the number of features $n$ is very large (e.g., $n > 10,000$).*
