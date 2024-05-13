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

errors = []

# Fitting the model and calculating the reconstruction error for the specific number of nodes
for nodes in [10, 100, 250]:
    model = build_autoencoder(nodes)
    model.fit(X_train, X_train,
              epochs=100,
              batch_size=256,
              validation_data=(X_val, X_val),
              callbacks=[early_stopping])
    test_loss = model.evaluate(X_test, X_test)
    errors.append(test_loss)
    print(f'Latent dimension {nodes} - Reconstruction Error: {test_loss}')

# Outputs
print("Reconstruction errors for different latent space sizes:")
print("10 nodes:", errors[0])
print("100 nodes:", errors[1])
print("250 nodes:", errors[2])

# 5
# Comments:


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
