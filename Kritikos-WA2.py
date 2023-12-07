import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# ------ Exercise 1 ------
print("------ Exercise 1 ------")

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

# Console output:
# Lasso coefficients:[ 1.99505138e+00  1.00934772e+00  7.17664501e-05 -8.06672247e-04]

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

# Console output:
# Ridge coefficients: [[ 2.00219889e+00  1.01185098e+00 -3.27195639e-04 -9.14443840e-04]]

# The first and second coefficients are approximately 2.002 and 1.012 which are close to
# the original coefficients for the linear and quadratic terms. The third and fourth
# coefficients are non-zero but smaller compared to the non-regularized case.

# ------ Exercise 2 ------
print("------ Exercise 2 ------")

# 1
# Loading the data and keeping only the data we need
iris = datasets.load_iris()
X = iris.data[:, :2]
y = np.where(iris.target == 0, 0, 1)  # 0 for Iris-Setosa, 1 for Iris-Virginica

# 2
# Visualizing the data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Length vs Sepal Width')
plt.show()

# 3
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training SVM classifiers with different values of C using GridSearchCV
param_grid = {'C': [10, 100]}
grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 4
# Getting the best C value from the grid search
best_C = grid_search.best_params_['C']
print(f"Best value of C = {best_C}")
# Console output: Best value of C = 100

# The grid search suggests that the optimal value of C is 100 for this specific dataset, and
# it leads to a more reliable model. This indicates that a stronger regularization is
# beneficial for capturing the underlying patterns in the data.

# 5
# Training the SVM classifier again with the best C value
clf = SVC(kernel='linear', C=best_C)
clf.fit(X_train, y_train)
# Testing the accuracy of C to be 100% sure about the previous question (4)
print(f"Accuracy for C = {best_C}: {accuracy_score(y_test, clf.predict(X_test))}")
# Console output:
# Accuracy for C = 100: 1.0

# Visualizing the decision boundary
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title(f'SVM Classifier (C={best_C})')

# Plotting decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
            linestyles=['--', '-', '--'])
plt.show()

# Training a Logistic Regression model
lr_model = make_pipeline(StandardScaler(), LogisticRegression())
lr_model.fit(X, y)

# 6
# Making contour plot for Logistic Regression
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
Z = lr_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Logistic Regression Contour Plot')
plt.show()

# 7
# Predicting probability for a sample
sample = np.array([[5.5, 3.25]])
sample_prob = lr_model.predict_proba(sample)[:, 1]
print(f"Probability of being Iris-Setosa: {sample_prob[0]:.4f}")
