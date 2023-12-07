import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline

# Code given by the exercise
np.random.seed(40)
m = 1000
X = 10 * np.random.rand(m, 1) - 5
y = 2*X + X**2 + np.random.randn(m, 1)

# 1
# Making the initial polynomial
poly_features = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly_features.fit_transform(X)

# 2
# Training the Lasso with alpha 0.01
lasso_model = make_pipeline(PolynomialFeatures(degree=4, include_bias=False), Lasso(alpha=0.01))
lasso_model.fit(X, y)

# 3
# Coefficients for Lasso model
lasso_coefficients = lasso_model.named_steps['lasso'].coef_
print("Lasso coefficients:", lasso_coefficients)

# The first and second coefficients are approximately 1.995 and 1.009 which are close to
# the linear term (2*X) and the quadratic term (X^2) respectively. The third and fourth
# coefficients are very close to zero, which means that Lasso regularization has
# succeeded in setting these coefficients to zero.

# 4
# Training the Ridge with alpha 0.01
ridge_model = make_pipeline(PolynomialFeatures(degree=4, include_bias=False), Ridge(alpha=0.01))
ridge_model.fit(X, y)

# 5
# Coefficients for Ridge model
ridge_coefficients = ridge_model.named_steps['ridge'].coef_
print("Ridge coefficients:", ridge_coefficients)

# The first and second coefficients are approximately 2.002 and 1.012 which are close to
# the original coefficients for the linear and quadratic terms. The third and fourth
# coefficients are non-zero but smaller compared to the non-regularized case.
