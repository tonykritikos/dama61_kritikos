import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

print("----- Problem 1 -----")

# 1
# Loading MNIST dataset & training and test sets
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Normalizing pixel values to be between 0 and 1
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Splitting data into training & validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=1/6, random_state=42)


# 2
# Building the autoencoder
def build_autoencoder(nodes):
    autoencoder_model = Sequential([
        Dense(nodes, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(X_train.shape[1], activation='sigmoid')
    ])
    autoencoder_model.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder_model


# 3 & 4
# Defining the early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fitting the model and calculating the reconstruction error for the specific number of nodes
for nodes in [10, 100, 250]:
    model = build_autoencoder(nodes)
    model.fit(X_train, X_train,
              epochs=100,
              batch_size=256,
              validation_data=(X_val, X_val),
              callbacks=[early_stopping])
    test_loss = model.evaluate(X_test, X_test)
    # output
    print(f'Latent dimension {nodes} - Reconstruction Error: {test_loss}')

# 5
# Console output:
#           Latent dimension 10 - Reconstruction Error: 0.0014080990804359317
#           Latent dimension 100 - Reconstruction Error: 0.00036464014556258917
#           Latent dimension 250 - Reconstruction Error: 5.670843893312849e-05

# Comments:
#         From the results we can see that as the number of nodes in the latent space increases, the reconstruction
#         error decreases. More specifically, with latent dimension number 10 the autoencoder struggles more to capture
#         the essential features of the data, leading to a higher reconstruction error. Increasing the latent dimension
#         to 100 provides a substantial improvement in the reconstruction error. This indicates that the model can now
#         encode more information and reconstruct the images more accurately. When increasing the dimension to 250
#         there is even more of an improvement (the readable number is close to 0.0000567) but not as significant as the
#         10 to 100 dimension improvement. The 250 dimension model is the most accurate of the three, but when taking
#         into consideration the runtime and performance the 100 dimension one is good enough for the task given.



print("----- Problem 2 -----")

# 1
# Generating time series data
t = np.linspace(0, 14 * np.pi, 10000)
original_data = np.cos(t)
# 2
# Adding Gaussian noise to the data
noise = np.random.normal(0, 0.05, size=original_data.shape)
noisy_data = original_data + noise

# Splitting the data into training and test sets
train_size = int(len(noisy_data) * 0.9)
train_data, test_data = noisy_data[:train_size], noisy_data[train_size:]


# 3
# Building and training the autoencoder
def build_and_train_autoencoder(nodes):
    autoencoder_model = Sequential([
        Dense(100, input_shape=(1,), activation='relu'),
        Dense(nodes, activation='relu'),
        Dense(100, activation='relu'),
        Dense(1, activation='linear')
    ])
    autoencoder_model.compile(optimizer='adam', loss='mse')
    autoencoder_model.fit(train_data.reshape(-1, 1), train_data.reshape(-1, 1),
              epochs=20, batch_size=100)
    return autoencoder_model


# 4
# Testing different latent space dimensions
for nodes in [1, 4, 20]:
    autoencoder = build_and_train_autoencoder(nodes)
    # Predict on the test set
    predicted = autoencoder.predict(test_data.reshape(-1, 1)).flatten()
    plt.figure(figsize=(10, 4))
    plt.plot(t[train_size:], test_data, label='Noisy Test Data')
    plt.plot(t[train_size:], predicted, label='Denoised by Autoencoder')
    plt.title(f'Latent Space Size: {nodes}')
    plt.legend()
    plt.show()

# 5
# Comments:
#         All autoencoders, regardless of latent space size, manage to reduce most of the noise in the test data.
#         The autoencoder with a latent space of 1 node struggles to capture the essential features of the original
#         signal as seen by the plot. The latent space of 4 nodes is sufficient to effectively denoise the signal
#         without significant loss of information. Increasing the size to 20 nodes does not visibly improve the
#         quality of the denoised signal. Given that, it implies that a smaller latent space (close to or equal to 4
#         nodes) is adequate. Using fewer nodes reduces the model complexity and computational requirements, making
#         the autoencoder more efficient without sacrificing performance.
