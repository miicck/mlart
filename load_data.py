import os
import random

import PIL.Image
import numpy as np
from PIL import Image
from typing import Tuple


def colorize_filter(image: np.ndarray):
    for i in range(3):
        image[:, :, i] += random.random()
    image -= np.floor(image)
    return image


def is_image(filename: str):
    return filename.endswith(".jpg")


def resize_numpy_array(input: np.ndarray, out_size: Tuple[int]) -> np.ndarray:
    img = np.array(input).astype(float)
    img *= 256 / max(img.flat)
    img = img.astype(np.uint8)
    img = PIL.Image.fromarray(img)
    return np.array(img.resize(out_size))


def load_data(directory: str,
              input_shape: Tuple[int, int, int],
              output_shape: Tuple[int, int, int],
              max_count: int = None,
              data_filter=None,
              input_filter=None,
              output_filter=None):
    assert input_shape[2] == 3
    assert output_shape[2] == 3

    # Get data paths
    data = [os.path.join(directory, x) for x in os.listdir(directory) if is_image(x)]
    random.shuffle(data)
    if max_count is not None:
        data = data[:max_count]

    # Initilize data arrays
    in_data = np.zeros((len(data), *input_shape))
    out_data = np.zeros((len(data), *output_shape))

    for i, filename in enumerate(data):
        try:
            img = Image.open(filename)
        except:
            continue

        if data_filter is not None:
            img = np.array(img.resize(img.size))
            img = img.astype(float)
            img /= max(img.flat)
            img = data_filter(img) * 256
            img = img.astype(np.uint8)
            img = PIL.Image.fromarray(img)

        # Save filtered image as input data
        in_data[i] = np.array(img.resize(input_shape[:2]))[:, :, :3]
        in_data[i] /= max(in_data[i].flat)
        if input_filter is not None:
            in_data[i] = input_filter(in_data[i])

        # Save raw image as output data
        out_data[i] = np.array(img.resize(output_shape[:2]))[:, :, :3]
        out_data[i] /= max(out_data[i].flat)
        if output_filter is not None:
            out_data[i] = output_filter(out_data[i])

        print(f"\rLoaded {i + 1}/{len(data)}           ", end="")
    print()

    return in_data, out_data
