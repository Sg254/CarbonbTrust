
# mnist_cnn_fixed.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and preprocess
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Ensure float32 and normalized, add channel dim
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# Add channel dimension: (N, 28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# Optionally augment: simple example using tf.data (flip/rotate isn't great for digits, but small shifts help)
def augment(image, label):
    image = tf.image.random_translation(image, [-2, 2], [-2, 2])  # small shift (TF >= 2.10)
    return image, label

batch_size = 128
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 2. Model
def make_model():
    inputs = keras.Input(shape=(28,28,1))
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(10, activation="softmax")(x)  # 10 classes
    model = keras.Model(inputs, outputs)
    return model

model = make_model()
model.summary()

# 3. Compile: use SparseCategoricalCrossentropy for integer labels
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

# 4. Train (with early stopping)
early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(train_ds, validation_data=test_ds, epochs=20, callbacks=[early])

# 5. Evaluate
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")

# 6. Visualize predictions on 5 samples
num_samples = 5
indices = np.random.choice(len(x_test), num_samples, replace=False)
samples = x_test[indices]
labels  = y_test[indices]
probs = model.predict(samples)
preds = np.argmax(probs, axis=-1)

plt.figure(figsize=(12,3))
for i in range(num_samples):
    plt.subplot(1, num_samples, i+1)
    plt.imshow(samples[i].squeeze(), cmap="gray")
    plt.title(f"GT:{labels[i]} Pred:{preds[i]}\nP:{probs[i][preds[i]]:.2f}")
    plt.axis("off")
plt.show()
