import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Loading Iris dataset
iris = datasets.load_iris()
X = iris.data[:, (2, 3)]  # Only keep the petal length and width
y = iris.target

# Filtering to keep only Iris-Versicolor and Iris-Virginica
versicolor_or_virginica = (y == 1) | (y == 2)
X = X[versicolor_or_virginica]
y = y[versicolor_or_virginica]

# Converting labels to 0 and 1 for binary classification
y = y - 1

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualizing the data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.title("Iris-Versicolor and Iris-Virginica")
plt.show()

# Training SVM classifiers with different values of C (I chose 10, 50, 100)
C_values = [10, 50, 100]
svm_models = [make_pipeline(StandardScaler(), SVC(kernel="linear", C=C, probability=True)) for C in C_values]

# Fitting the models
for model in svm_models:
    model.fit(X_train, y_train)

# Evaluating the models
for i, model in enumerate(svm_models):
    print(f"SVM Model with C={C_values[i]}: Train accuracy={model.score(X_train, y_train)}, Test accuracy={model.score(X_test, y_test)}")

# Choosing the best model (Choosing 0 for the first-one since for C=10 we have the best results)
best_svm_model = svm_models[0]

# Given function
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    margin = 1 / np.linalg.norm(w)
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    svs = svm_clf.support_vectors_
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#AAA', zorder=-1)

# Extracting the trained SVM model from the pipeline
svm = best_svm_model.named_steps["svc"]
scaler = best_svm_model.named_steps["standardscaler"]

# Scaling the whole dataset using the same scaler
X_scaled = scaler.transform(X)

# Plotting the decision boundary
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
plot_svc_decision_boundary(svm, X_scaled[:, 0].min(), X_scaled[:, 0].max())
plt.xlabel("Petal length (scaled)")
plt.ylabel("Petal width (scaled)")
plt.title("SVM Decision Boundary (scaled)")
plt.show()

# Training Logistic Regression
log_reg = make_pipeline(StandardScaler(), LogisticRegression())
log_reg.fit(X_train, y_train)

# Training Decision Tree
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_train, y_train)

# Training AdaBoost
ada_clf = AdaBoostClassifier(n_estimators=10)
ada_clf.fit(X_train, y_train)

# Evaluating all models
models = [best_svm_model, log_reg, tree_clf, ada_clf]
model_names = ["SVM", "Logistic Regression", "Decision Tree", "AdaBoost"]

for model, name in zip(models, model_names):
    print(f"{name}: Train accuracy={model.score(X_train, y_train)}, Test accuracy={model.score(X_test, y_test)}")

# Sampling features
sample = np.array([[5.0, 1.5]])
sample_scaled = scaler.transform(sample)

# Calculating probabilities
for model, name in zip(models, model_names):
    if name == "SVM":
        prob = model.named_steps["svc"].predict_proba(sample_scaled)[0, 0]
    else:
        prob = model.predict_proba(sample_scaled)[0, 0]
    print(f"Probability of being Iris Versicolor for {name}: {prob:.4f}")
