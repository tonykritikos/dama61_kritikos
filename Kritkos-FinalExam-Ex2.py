import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loading and splitting the MNIST dataset
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, test_size=1/7, random_state=42)

# Normalizing the data
x_train = x_train / 255.0
x_valid = x_valid / 255.0
x_test = x_test / 255.0

# Flattening the data for Random Forest
x_train_flatten = x_train.reshape(x_train.shape[0], -1)
x_valid_flatten = x_valid.reshape(x_valid.shape[0], -1)
x_test_flatten = x_test.reshape(x_test.shape[0], -1)

# Training Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
rf_clf.fit(x_train_flatten, y_train)

# Calculating the score and confusion matrix for Random Forest
rf_valid_score = rf_clf.score(x_valid_flatten, y_valid)
rf_test_score = rf_clf.score(x_test_flatten, y_test)
y_pred_rf = rf_clf.predict(x_test_flatten)
rf_conf_matrix = confusion_matrix(y_test, y_pred_rf)

print(f"Random Forest Validation Score: {rf_valid_score}")
print(f"Random Forest Test Score: {rf_test_score}")
print(f"Random Forest Confusion Matrix:\n{rf_conf_matrix}")

# Turning values to categorical
y_train_cat = to_categorical(y_train)
y_valid_cat = to_categorical(y_valid)
y_test_cat = to_categorical(y_test)

# Building the CNN model
cnn_model = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compiling the CNN model
cnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the CNN model with early stopping
early_stopping = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)
cnn_history = cnn_model.fit(x_train[..., np.newaxis], y_train_cat,
                            epochs=100,
                            validation_data=(x_valid[..., np.newaxis], y_valid_cat),
                            callbacks=[early_stopping])

# Plotting the history of loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['loss'], label='Training Loss')
plt.plot(cnn_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss History')

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['accuracy'], label='Training Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy History')

plt.show()

# Evaluating the CNN model and calculating the confusion matrix
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test[..., np.newaxis], y_test_cat)
y_pred_cnn = cnn_model.predict(x_test[..., np.newaxis])
y_pred_cnn_classes = np.argmax(y_pred_cnn, axis=1)
cnn_conf_matrix = confusion_matrix(y_test, y_pred_cnn_classes)

print(f"CNN Test Accuracy: {cnn_test_acc}")
print(f"CNN Confusion Matrix:\n{cnn_conf_matrix}")

# Making visual comparison of confusion matrices
print(f"Random Forest Confusion Matrix:\n{rf_conf_matrix}")
print(f"CNN Confusion Matrix:\n{cnn_conf_matrix}")
