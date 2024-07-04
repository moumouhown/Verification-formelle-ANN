import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Preprocess data
train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images = test_images.reshape(-1, 28 * 28) / 255.0

# One-hot encode labels
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Define the model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(512, activation="relu", input_shape=(28 * 28,)),
  tf.keras.layers.Dense(256, activation="relu"),
  tf.keras.layers.Dense(10, activation="softmax")  # Output layer with 10 units for 10 digits
])

# Compile the model
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the model (adjust the number of epochs as needed)
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the trained model
model.save("MNIST-NN_Model.keras")

# Export the TensorFlow model as a saved model
tf.saved_model.save(model, "mnist_model")
