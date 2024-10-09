import tensorflow as tf
from tensorflow.keras import layers, models

# Fully connected feedforward neural network
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Input layer
    layers.Dense(128, activation='relu'),  # Hidden layer 1
    layers.Dense(64, activation='relu'),   # Hidden layer 2
    layers.Dense(10, activation='softmax') # Output layer
])

model.summary()