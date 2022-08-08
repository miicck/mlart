import random
from ml.load_data import load_data, colorize_filter, resize_numpy_array
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from typing import Iterable


def plot_predictions(model, in_data: np.ndarray, training_data: np.ndarray = None):
    prediction = model.predict(in_data)
    assert len(prediction) == len(in_data)

    cols = 2 if training_data is None else 3

    for i in range(len(in_data)):
        # Plot the input data
        plt.subplot(len(in_data), cols, i * cols + 1)
        plt.imshow(in_data[i])

        # Plot the prediction
        plt.subplot(len(in_data), cols, i * cols + 2)
        plt.imshow(prediction[i])

        if training_data is None:
            continue  # No training data provided

        # Plot training data
        plt.subplot(len(in_data), cols, i * cols + 3)
        plt.imshow(training_data[i])

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)


def gen_rectangles(shape: Iterable[int], noise_strength: float = 0.0):
    result = np.zeros(shape)

    for i in range(shape[0]):

        for j in range(100):
            x_size, y_size = np.random.random(2)
            x_size = int((x_size * 0.4 + 0.1) * shape[1])
            y_size = int((y_size * 0.4 + 0.1) * shape[2])

            x, y = np.random.random(2)
            x = int(x * (shape[1] - x_size))
            y = int(y * (shape[2] - y_size))

            result[i, x:x + x_size, y:y + y_size, :] = np.random.random(3)

    result = result * (1 - noise_strength) + np.random.random(result.shape) * noise_strength
    return result


def predict_composite(model, image: np.ndarray) -> np.ndarray:
    # Input/output shapes of the model
    in_shape = model.layers[0].input_shape[1:4]
    out_shape = model.layers[-1].output_shape[1:4]

    x_step = in_shape[0]
    y_step = in_shape[1]

    count = int(max(image.shape[0] / x_step, image.shape[1] / y_step))

    image_divided = np.zeros((count * count, *in_shape))
    for x in range(count):
        for y in range(count):
            image_divided[x + y * count] = image[x * x_step:(x + 1) * x_step, y * y_step:(y + 1) * y_step]

    predict = model.predict(image_divided)
    composite = np.zeros((predict.shape[1] * count, predict.shape[2] * count, 3))

    x_step = predict.shape[1]
    y_step = predict.shape[2]

    for x in range(count):
        for y in range(count):
            composite[x * x_step:(x + 1) * x_step, y * y_step:(y + 1) * y_step] = predict[x + y * count]

    return composite


def predict_average_composite(model, image: np.ndarray) -> np.ndarray:
    # Input/output shapes of the model
    in_shape = model.layers[0].input_shape[1:4]
    out_shape = model.layers[-1].output_shape[1:4]

    # Ratio of output/input sizes
    shape_ratio = (out_shape[0] / in_shape[0], out_shape[1] / in_shape[1])

    # The resulting composite image shape
    result_shape = (image.shape[0] * shape_ratio[0], image.shape[1] * shape_ratio[1])
    result_shape = [int(x + 0.5) for x in result_shape]
    result_shape.append(3)

    # The resulting composite image
    result = np.zeros(result_shape)

    # Steps to move the stencil over the source image
    x_steps = image.shape[0] - in_shape[0]
    y_steps = image.shape[1] - in_shape[1]

    if False:
        for n in range(100):
            x = random.randrange(0, x_steps)
            y = random.randrange(0, y_steps)
            subimage = np.zeros((1, *in_shape))
            subimage[0] = image[x:x + in_shape[0], y:y + in_shape[1], :]
            predict = model.predict(subimage)[0]
            x_out = int((x + random.random()) * shape_ratio[0])
            y_out = int((y + random.random()) * shape_ratio[1])
            result[x_out:x_out + out_shape[0], y_out:y_out + out_shape[1], :] += predict

    else:
        # Move stencil horizontally
        for x in range(0, x_steps):

            # Move stencil vertically + store snapshot images at each vertical position
            subimage_row = np.zeros((y_steps, *in_shape))
            for y in range(0, y_steps):
                subimage_row[y] = image[x: x + in_shape[0], y:y + in_shape[1], :]

            # Make predictions from snapshot images
            row_predicitons = model.predict(subimage_row)

            # Avearage predictions into result image
            for y in range(0, y_steps):
                x_out = int((x + random.random()) * shape_ratio[0])
                y_out = int((y + random.random()) * shape_ratio[1])
                result[x_out:x_out + out_shape[0], y_out:y_out + out_shape[1], :] += row_predicitons[y]

            print(f"{x}/{x_steps} {row_predicitons.shape}")

    result /= max(result.flat)
    return result


assert os.path.isdir(sys.argv[1])
assert os.path.isdir(sys.argv[2])
model_to_load = sys.argv[2]

model = keras.models.load_model(model_to_load)
in_shape = model.layers[0].input_shape[1:4]
out_shape = model.layers[-1].output_shape[1:4]
print("Loaded model with\n"
      f"     input shape: {in_shape}\n"
      f"    output shape: {out_shape}")

# Plot some predictions for loaded images
plt.figure()
x_data, y_data = load_data(sys.argv[1], in_shape, out_shape, max_count=4)
plot_predictions(model, x_data, training_data=y_data)

# Plot some predictions for noise input
plt.figure()
noise_data = np.random.random((4, *in_shape))
plot_predictions(model, noise_data)

# Plot some predictions for random rectangles
plt.figure()
squares_data = gen_rectangles((4, *in_shape))
plot_predictions(model, squares_data)

# Plot some composite images, using various compositing schemes
for compositing_method in [predict_composite, predict_average_composite]:
    plt.figure()

    plt.subplot(321)
    img = gen_rectangles((1, in_shape[0] * 4, in_shape[1] * 4, 3))[0]
    plt.imshow(img)
    plt.subplot(322)
    plt.imshow(compositing_method(model, img))

    plt.subplot(323)
    img = gen_rectangles((1, in_shape[0] * 4, in_shape[1] * 4, 3), noise_strength=0.5)[0]
    plt.imshow(img)
    plt.subplot(324)
    plt.imshow(compositing_method(model, img))

    plt.subplot(325)
    img = model.predict(np.random.random((1, *in_shape)))[0]
    img = resize_numpy_array(img, (in_shape[0] * 4, in_shape[1] * 4))
    plt.imshow(img)
    plt.subplot(326)
    plt.imshow(compositing_method(model, img))

plt.show()
