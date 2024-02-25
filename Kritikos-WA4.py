import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist

# Loading MNIST dataset
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Normalizing pixel values to be between 0 and 1
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flattening the images
X_train_full = X_train_full.reshape((-1, 28*28))
X_test = X_test.reshape((-1, 28*28))

# Splitting dataset into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=1/7, random_state=42)

# Defining the 50 node model
model_50 = Sequential([
    Dense(50, activation='relu', input_shape=(784,)),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(10, activation='softmax')
])

# Defining the 200 node model
model_200 = Sequential([
    Dense(200, activation='relu', input_shape=(784,)),
    Dropout(0.5),  # Dropout regularization with dropout rate of 0.5
    Dense(200, activation='relu'),
    Dropout(0.5),  # Dropout regularization with dropout rate of 0.5
    Dense(200, activation='relu'),
    Dropout(0.5),  # Dropout regularization with dropout rate of 0.5
    Dense(10, activation='softmax')
])

# Compiling the models
model_50.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_200.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Calculating the number of trainable parameters
model_50.summary()
model_200.summary()

# Python console output:
#
# Model: "sequential"
# _________________________________________________________________
# Layer(type)       Output Shape        Param  #
# == == == == == == == == == == == == == == == == == == == == == ==
# dense(Dense)      (None, 50)          39250
# dense_1(Dense)    (None, 50)          2550
# dense_2(Dense)    (None, 50)          2550
# dense_3(Dense)    (None, 10)          510
# == == == == == == == == == == == == == == == == == == == == == ==
# Total params: 44860(175.23 KB)
# Trainable params: 44860(175.23 KB)
# Non - trainable params: 0(0.00 Byte)
# _________________________________________________________________
#
#
# Model: "sequential_1"
# _________________________________________________________________
# Layer(type)       Output Shape        Param  #
# == == == == == == == == == == == == == == == == == == == == == ==
# dense_4(Dense)    (None, 200)         157000
# dropout(Dropout)  (None, 200)         0
# dense_5(Dense)    (None, 200)         40200
# dropout_1(Dropout)(None, 200)         0
# dense_6(Dense)    (None, 200)         40200
# dropout_2(Dropout)(None, 200)         0
# dense_7(Dense)    (None, 10)          2010
# == == == == == == == == == == == == == == == == == == == == == ==
# Total params: 239410(935.20 KB)
# Trainable params: 239410(935.20 KB)
# Non - trainable params: 0(0.00 Byte)
# _________________________________________________________________

# Defining early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Fitting the models
history_50 = model_50.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping])
history_200 = model_200.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping])

for nodes, history, model in [('50', history_50, model_50), ('200', history_200, model_200)]:
    # Plotting each model's training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Loss for Model with {} Nodes".format(nodes))
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Accuracy for Model with {} Nodes".format(nodes))
    plt.legend()
    plt.show()

    # Evaluating each model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Loss for {} nodes: {}".format(nodes, test_loss))
    print("Test Accuracy for {} nodes: {}".format(nodes, test_accuracy))

    # Python console output:
    #
    # Test Loss for 50 nodes: 0.10910116136074066
    # Test Accuracy for 50 nodes: 0.9696999788284302
    #
    # Test Loss for 200 nodes: 0.09260601550340652
    # Test Accuracy for 200 nodes: 0.9761000275611877
