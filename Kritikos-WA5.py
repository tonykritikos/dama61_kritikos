import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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
