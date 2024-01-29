import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ------ Exercise 1 ------
print("------ Exercise 1 ------")

# Setting the random state
random_state = 42

# Loading the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
X, y = mnist.data.values.astype(np.float32), mnist.target.astype(int)  # Using astype() and np.float32 to eliminate IDE errors

# 1
# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=random_state)

# 2
# Preprocessing for better runtime
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# 3
# Using PCA to reduce dimensions and preserve 90% of the training set's variance
pca = PCA(n_components=0.90, random_state=random_state)
X_train_pca = pca.fit_transform(X_train_preprocessed)
X_test_pca = pca.transform(X_test_preprocessed)

# Initializing classifiers
decision_tree = DecisionTreeClassifier(max_depth=10, random_state=random_state)
random_forest = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1) # n_jobs added for better runtime
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
# Console output:
#             Decision Tree Score: 0.7932
#             Random Forest Score: 0.9329
#             AdaBoost Score: 0.7
#             Linear SVC Score: 0.8996
#             Logistic Regression Score: 0.9216

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
# Console output:
#             Stacking Classifier Score: 0.9316

# 5
# The Stacking Classifier showed a performance improvement compared to individual classifiers, with a score of 0.9316.
# This suggests that combining diverse classifiers using ensemble techniques can yield enhanced predictive capabilities.
# The Random Forest final estimator in the stacking ensemble played a crucial role in achieving this improved performance.
# The extended runtime could impact the practical applicability of the model or the machine that is running on, especially
# in scenarios where quick predictions are essential. Consideration should be given to balancing computational efficiency
# with model performance, and further exploration of alternative models or optimization techniques may be needed.

# ------ Exercise 2 ------
print("      ")
print("------ Exercise 2 ------")

# Loading the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data.astype('float32'), mnist.target.astype('int')

# 1
# Filtering digits 7, 8 and 9
mask = (y == 7) | (y == 8) | (y == 9)
X_filtered = X[mask]
y_filtered = y[mask]

# 2
# Transforming the data by applying PCA and keeping only the first two components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_filtered)

# 3
# Visualizing the data
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_filtered, cmap='jet')
plt.title('PCA of MNIST digits 7, 8, and 9')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar()
plt.show()

# 4

# (5) Initializing attributes for best score and best number of clusters
best_score = 0
best_n_clusters = 0

# Training K-Means with 2 to 10 clusters and calculate silhouette scores
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X_pca)
    silhouette_avg = silhouette_score(X_pca, labels)
    print(f'Number of clusters: {n_clusters}, Silhouette Score: {silhouette_avg}')

    # (5) Deciding the best score and number of clusters
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_n_clusters = n_clusters
# Console output:
#             Number of clusters: 2, Silhouette Score: 0.3727215528488159
#             Number of clusters: 3, Silhouette Score: 0.4312571585178375
#             Number of clusters: 4, Silhouette Score: 0.3685321509838104
#             Number of clusters: 5, Silhouette Score: 0.3844033479690552
#             Number of clusters: 6, Silhouette Score: 0.38583704829216003
#             Number of clusters: 7, Silhouette Score: 0.38510382175445557
#             Number of clusters: 8, Silhouette Score: 0.3736627995967865
#             Number of clusters: 9, Silhouette Score: 0.3698442578315735
#             Number of clusters: 10, Silhouette Score: 0.362096905708313

# 5
print(f'Best number of clusters: {best_n_clusters}')
# Console output:
#             Best number of clusters: 3

# The best number of clusters is 3 which agrees with the number of digits in the data,
# so this number will be used for the next question

# 6
# Training the best K-Means model
best_kmeans = KMeans(n_clusters=best_n_clusters, random_state=random_state)
best_labels = best_kmeans.fit_predict(X_pca)

# Finding the center of each cluster
cluster_centers = pca.inverse_transform(best_kmeans.cluster_centers_)

# Plotting representative images of each cluster
fig, ax = plt.subplots(1, best_n_clusters, figsize=(10, 4))
for i in range(best_n_clusters):
    ax[i].imshow(cluster_centers[i].reshape(28, 28), cmap='gray')
    ax[i].set_title(f'Cluster {i+1}')
    ax[i].axis('off')

plt.show()
