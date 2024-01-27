import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_openml


# ------ Exercise 1 ------
print("------ Exercise 1 ------")

# Setting the random state
random_state = 42

# Loading the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')  # Set extra attributes due to IDE errors
X, y = mnist.data.values.astype(np.float32), mnist.target.astype(int)

# 1
# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=random_state)

# 2
# Using PCA to reduce dimensions and preserve 90% of the training set's variance
pca = PCA(n_components=0.90, random_state=random_state)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 3
# Initializing classifiers
decision_tree = DecisionTreeClassifier(max_depth=10, random_state=random_state)
random_forest = RandomForestClassifier(n_estimators=50, random_state=random_state)
adaboost = AdaBoostClassifier(n_estimators=50, random_state=random_state)
linear_svc = make_pipeline(LinearSVC(max_iter=500, random_state=random_state))
logistic_regression = LogisticRegression(max_iter=500, random_state=random_state)

# Training classifiers on the reduced training set
decision_tree.fit(X_train_pca, y_train)
random_forest.fit(X_train_pca, y_train)
adaboost.fit(X_train_pca, y_train)
linear_svc.fit(X_train_pca, y_train)
logistic_regression.fit(X_train_pca, y_train)

# Evaluating score for individual classifiers on the test set
dt_score = decision_tree.score(X_test_pca, y_test)
rf_score = random_forest.score(X_test_pca, y_test)
ab_score = adaboost.score(X_test_pca, y_test)
svc_score = linear_svc.score(X_test_pca, y_test)
lr_score = logistic_regression.score(X_test_pca, y_test)

print("Decision Tree Score:", dt_score)
print("Random Forest Score:", rf_score)
print("AdaBoost Score:", ab_score)
print("Linear SVC Score:", svc_score)
print("Logistic Regression Score:", lr_score)

# 4
# Combining all the classifiers
stacking_classifier = StackingClassifier(
    estimators=[
        ('decision_tree', decision_tree),
        ('adaboost', adaboost),
        ('linear_svc', linear_svc),
        ('logistic_regression', logistic_regression)],
    final_estimator=random_forest,
    cv=3
)

# Training the Stacking Classifier on the reduced training set
stacking_classifier.fit(X_train_pca, y_train)

# Evaluating the Stacking Classifier on the test set
stacking_score = stacking_classifier.score(X_test_pca, y_test)

print("Stacking Classifier Score:", stacking_score)

# 5
# Comment
