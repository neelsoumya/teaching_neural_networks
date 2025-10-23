"""
simple_cnn.py

A minimal, well-commented example of a Convolutional Neural Network (CNN)
using Keras (tf.keras). Trains on MNIST handwritten digits so students can
see how convolutions + pooling + dense layers work together.

Run: python simple_cnn.py
Or open and run the cells in a Jupyter notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------
# 1) Load a tiny dataset
# --------------------------
# MNIST: 28x28 grayscale images of handwritten digits (0-9)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# For CNNs we want a 4D tensor: (batch, height, width, channels).
# MNIST is grayscale so channels = 1.
x_train = x_train.astype("float32") / 255.0  # scale pixels to [0,1]
x_test = x_test.astype("float32") / 255.0

# Add the "channels" dimension
x_train = np.expand_dims(x_train, -1)  # shape -> (num_samples, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)

# --------------------------
# 2) Build a tiny CNN model
# --------------------------
# We'll use:
# - Conv2D: applies small filters (kernels) sliding over the image
# - MaxPooling2D: reduces spatial size, keeps strongest features
# - Flatten: converts 2D feature maps to 1D vector
# - Dense: fully connected layer for classification

model = keras.Sequential(
    [
        # First convolutional layer:
        #  - 8 filters (small number for simplicity)
        #  - kernel_size=3 (3x3 filter)
        #  - activation='relu' (simple non-linearity)
        layers.Conv2D(8, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
        # Pooling layer: reduces width & height by half (28->14)
        layers.MaxPooling2D(pool_size=2),

        # Second convolutional layer to learn higher-level patterns
        layers.Conv2D(16, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),  # 14->7 -> after this we have small feature maps

        # Flatten feature maps into a vector for the Dense layer
        layers.Flatten(),

        # A small dense layer to mix features
        layers.Dense(32, activation="relu"),

        # Output layer: 10 units (one per digit) with softmax to give probabilities
        layers.Dense(10, activation="softmax"),
    ]
)

# Show a summary of the model so students can see layer names and shapes
model.summary()

# --------------------------
# 3) Compile the model
# --------------------------
# - optimizer: how the model updates weights (Adam is a good default)
# - loss: how we measure error (sparse categorical because labels are integers)
# - metrics: what we report (accuracy)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# --------------------------
# 4) Train the model
# --------------------------
# For teaching/demo keep epochs small so it runs quickly.
# Increase epochs for better accuracy if desired.
history = model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128)

# --------------------------
# 5) Evaluate on test data
# --------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.3f}, Test loss: {test_loss:.3f}")

# --------------------------
# 6) Visualize predictions
# --------------------------
# Show a few test images with predicted and true labels.
num_display = 8
sample_images = x_test[:num_display]
sample_labels = y_test[:num_display]

pred_probs = model.predict(sample_images)          # probabilities for each class
pred_labels = np.argmax(pred_probs, axis=1)        # pick class with highest prob

plt.figure(figsize=(12, 3))
for i in range(num_display):
    plt.subplot(1, num_display, i + 1)
    image = sample_images[i].squeeze()  # remove channel dim for plotting
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.title(f"pred:{pred_labels[i]}\ntrue:{sample_labels[i]}")
plt.show()

# --------------------------
# 7) Short explanation for students (print)
# --------------------------
print("""
Quick recap (plain English):
- Convolution (Conv2D): small filters (e.g. 3x3) slide over the image and compute
  local features (edges, textures). Each filter produces a feature map.
- Pooling (MaxPooling2D): down-samples feature maps to make them smaller and
  keep the strongest signals — helps with translation invariance.
- Flatten -> Dense: takes the final feature maps and uses them to decide the class.
- Softmax output: converts the final numbers into probabilities over digits 0-9.
""")
