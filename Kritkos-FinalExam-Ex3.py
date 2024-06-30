import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Loading the MNIST dataset and filter the digits 1, 4, and 8
mnist = fetch_openml('mnist_784', version=1)
data = mnist['data']
target = mnist['target'].astype(int)
mask = np.isin(target, [1, 4, 8])
data = data[mask]
target = target[mask]

# Applying PCA
pca = PCA(0.80)
data_pca = pca.fit_transform(data)
data1 = data_pca

# Finding the number of principal components needed
n_components = pca.n_components_

# Building a stacked autoencoder
input_dim = data.shape[1]
encoding_dim = n_components

input_img = Input(shape=(input_dim,))
encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(256, activation='relu')(encoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Training the autoencoder
autoencoder.fit(data, data, epochs=20, batch_size=64, shuffle=True, validation_split=0.2, verbose=1)

# Getting the reduced dataset using the encoder
data2 = encoder.predict(data)

# Applying k-means clustering and evaluating using silhouette score
def evaluate_kmeans(data, k_range):
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(data)
        score = silhouette_score(data, clusters)
        scores.append(score)
    return scores

k_range = range(2, 11)
scores_data1 = evaluate_kmeans(data1, k_range)
scores_data2 = evaluate_kmeans(data2, k_range)

# Plotting silhouette scores
plt.figure(figsize=(12, 6))
plt.plot(k_range, scores_data1, label='PCA Reduced Data')
plt.plot(k_range, scores_data2, label='Autoencoder Reduced Data')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.legend()
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.show()

# Finding optimal number of clusters
optimal_clusters_data1 = k_range[np.argmax(scores_data1)]
optimal_clusters_data2 = k_range[np.argmax(scores_data2)]

print(f'Optimal number of clusters for PCA reduced data: {optimal_clusters_data1}')
print(f'Optimal number of clusters for Autoencoder reduced data: {optimal_clusters_data2}')
print(f'Actual number of classes: {len(np.unique(target))}')
