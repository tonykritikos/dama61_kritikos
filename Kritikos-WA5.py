import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Exercise 1

# Loading MNIST dataset & training and test sets
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Normalizing pixel values to be between 0 and 1
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 1
# Splitting data into training & validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=1/6, random_state=42)


# 2
# Defining the CNN model
def build_cnn_model(config):
    model = Sequential([
        Conv2D(config['conv1_filters'], (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(config['conv2_filters'], (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(config['conv3_filters'], (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model


# 6
# Defining configurations
config1 = {'conv1_filters': 32, 'conv2_filters': 64, 'conv3_filters': 64}
config2 = {'conv1_filters': 64, 'conv2_filters': 128, 'conv3_filters': 128}

# 3
# Compiling the models
model_config1 = build_cnn_model(config1)
model_config1.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

model_config2 = build_cnn_model(config2)
model_config2.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

# Defining early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 4
# Fitting the models
history_config1 = model_config1.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=100,
                                    validation_data=(X_val.reshape(-1, 28, 28, 1), y_val), callbacks=[early_stopping])

history_config2 = model_config2.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=100,
                                    validation_data=(X_val.reshape(-1, 28, 28, 1), y_val), callbacks=[early_stopping])

# 5
# Plotting training history for configuration 1
plt.plot(history_config1.history['loss'], label='Training Loss (Config 1)')
plt.plot(history_config1.history['val_loss'], label='Validation Loss (Config 1)')
plt.plot(history_config1.history['accuracy'], label='Training Accuracy (Config 1)')
plt.plot(history_config1.history['val_accuracy'], label='Validation Accuracy (Config 1)')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.title('Training History (Configuration 1)')
plt.legend()
plt.show()

# Plotting training history for configuration 2
plt.plot(history_config2.history['loss'], label='Training Loss (Config 2)')
plt.plot(history_config2.history['val_loss'], label='Validation Loss (Config 2)')
plt.plot(history_config2.history['accuracy'], label='Training Accuracy (Config 2)')
plt.plot(history_config2.history['val_accuracy'], label='Validation Accuracy (Config 2)')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.title('Training History (Configuration 2)')
plt.legend()
plt.show()

# 6
# Evaluating models on test set
test_loss_config1, test_acc_config1 = model_config1.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)
test_loss_config2, test_acc_config2 = model_config2.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)

print("Test accuracy for model configuration 1:", test_acc_config1)
print("Test accuracy for model configuration 2:", test_acc_config2)

# Console output:
#           Test accuracy for model configuration 1: 0.9904000163078308
#           Test accuracy for model configuration 2: 0.9911999702453613

# Comment:
#           The models trained with both configurations achieved high test accuracies of approximately 99%, indicating
#           that they generalize well to unseen data. Despite the observed trends of high training and validation
#           accuracies, the models did not suffer from significant overfitting, as evidenced by their strong performance
#           on the test set. These results suggest that the chosen architectures effectively captured the underlying
#           patterns in the MNIST dataset.


# Exercise 2

# Generating time series data
t = np.linspace(0, 14 * np.pi, 10000)
f = np.cos(t)

# 1
# Splitting data into training, validation and test sets
X_train, X_test, y_train, y_test = train_test_split(f[:-1], f[1:], test_size=0.2, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

# 2
# Function to convert time series data into supervised learning dataset
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)


window_size = 10
X_train_win10, y_train_win10 = create_dataset(X_train, window_size)
X_val_win10, y_val_win10 = create_dataset(X_val, window_size)
X_test_win10, y_test_win10 = create_dataset(X_test, window_size)

# 3
# Building LSTM model
model_win10 = Sequential([
    LSTM(100, input_shape=(window_size, 1)),
    Dense(1)
])

# 4
# Compiling the model
model_win10.compile(optimizer='adam', loss='mse')

# 5
# Fitting the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history_win10 = model_win10.fit(X_train_win10[..., np.newaxis], y_train_win10, epochs=100,
                                validation_data=(X_val_win10[..., np.newaxis], y_val_win10),
                                callbacks=[early_stopping])

# Plotting training and validation loss
plt.plot(history_win10.history['loss'], label='Training Loss')
plt.plot(history_win10.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (Window Size = 10)')
plt.legend()
plt.show()

# 7
# Making the same process with increased window size
window_size = 20
X_train_win20, y_train_win20 = create_dataset(X_train, window_size)
X_val_win20, y_val_win20 = create_dataset(X_val, window_size)
X_test_win20, y_test_win20 = create_dataset(X_test, window_size)

# Building LSTM model with the increased window size
model_win20 = Sequential([
    LSTM(100, input_shape=(window_size, 1)),
    Dense(1)
])

# Compiling the model with the increased window size
model_win20.compile(optimizer='adam', loss='mse')

# Fitting the new model with early stopping
history_win20 = model_win20.fit(X_train_win20[..., np.newaxis], y_train_win20, epochs=100,
                                validation_data=(X_val_win20[..., np.newaxis], y_val_win20),
                                callbacks=[early_stopping])

# Plotting training and validation loss for the new model
plt.plot(history_win20.history['loss'], label='Training Loss')
plt.plot(history_win20.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (Window Size = 20)')
plt.legend()
plt.show()

# Evaluating both models on test set
loss_win10 = model_win10.evaluate(X_test_win10[..., np.newaxis], y_test_win10)
loss_win20 = model_win20.evaluate(X_test_win20[..., np.newaxis], y_test_win20)
print("Test Loss (Window Size = 10): {:.8f}".format(loss_win10))
print("Test Loss (Window Size = 20): {:.8f}".format(loss_win20))

# Console output:
#           Test Loss (Window Size = 10): 0.00000103
#           Test Loss (Window Size = 20): 0.00000035

# Comment:
#           Both models exhibit a decrease in training loss, starting from around 0.019 for window size 10 and 0.009
#           for window size 20, dropping close to 0 within the first epoch. Despite both models achieving low training
#           loss values, the model with a window size of 20 performs slightly better on the test set, as evidenced by
#           its lower test loss compared to the model with a window size of 10. Increasing the window size from 10 to 20
#           results in a model that performs slightly better on the test set, indicating that a larger context window
#           allows the model to capture more relevant temporal dependencies in the data. It is essential to keep in mind
#           the test samples results, to know if the model overfits the data (which doesn’t as we see on the results),
#           but even though the larger window does perform better, the increase in accuracy is small enough to make it
#           not worth in cases that runtime is of importance.
