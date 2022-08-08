import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from ml.load_data import load_data, colorize_filter
import sys

assert os.path.isdir(sys.argv[1])

inp_shape = (8, 8, 3)

smooth_activation = "selu"
output_activation = "sigmoid"
sharp_activation = "leaky_relu"

model = keras.models.Sequential()
model.add(layers.UpSampling2D(size=(8, 8), input_shape=inp_shape))
model.add(layers.Conv2D(filters=3 * 3 * 3, kernel_size=(5, 5), activation=sharp_activation))
# model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(filters=3 * 3 * 3, kernel_size=(11, 11), activation=smooth_activation))
# model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(filters=3 * 3 * 3, kernel_size=(9, 9), activation=sharp_activation))
model.add(layers.UpSampling2D(size=(8, 8)))
model.add(layers.Conv2D(filters=3 * 3 * 3, kernel_size=(7, 7), activation=smooth_activation))
# model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(filters=3 * 3, kernel_size=(5, 5), activation=sharp_activation))
# model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(filters=3, kernel_size=(3, 3), activation=output_activation))

# Print info about the model
model.summary()
out_shape = model.layers[-1].output_shape[1:4]
print(f" Input shape: {inp_shape}")
print(f"Output shape: {out_shape}")

# Load input data
x_data, y_data = load_data(sys.argv[1], inp_shape, out_shape)

# Fit model
model.compile(optimizer="adam", loss="mse")
history = model.fit(x_data, y_data, epochs=1000)

# Remove old model, save new model
if os.path.isdir("model.save"):
    os.system("rm -r model.save")
model.save("model.save")

# Show training behaviour
if False:
    plt.plot(history.history['loss'], marker="+")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
