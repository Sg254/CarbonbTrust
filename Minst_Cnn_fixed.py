# BUGGY CODE - DO NOT USE
import tensorflow as tf
from tensorflow import keras

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)  # Bug 1: Missing activation
])

# Bug 2: Wrong loss function for the output
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # WRONG for multi-class
    metrics=['accuracy']
)

# Bug 3: Not normalizing data
model.fit(x_train, y_train, epochs=5)

# Bug 4: Wrong dimension for prediction
test_image = x_test[0]
prediction = model.predict(test_image)  # Missing batch dimension
FIXED CODE
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# FIX 1: Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),  # Add regularization
    # FIX 2: Add softmax activation for multi-class classification
    keras.layers.Dense(10, activation='softmax')
])

# FIX 3: Use correct loss function
# sparse_categorical_crossentropy for integer labels
# categorical_crossentropy for one-hot encoded labels
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Correct for integer labels
    metrics=['accuracy']
)

# Train model
history = model.fit(
    x_train, y_train, 
    epochs=5,
    batch_size=32,
    validation_split=0.2,  # Add validation
    verbose=1
)

# FIX 4: Add batch dimension for prediction
test_image = x_test[0]
test_image = np.expand_dims(test_image, axis=0)  # Shape: (1, 28, 28)
prediction = model.predict(test_image)
predicted_class = np.argmax(prediction[0])

print(f"Predicted class: {predicted_class}")
print(f"Actual class: {y_test[0]}")

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')
