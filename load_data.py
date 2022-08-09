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


def load_file(filename: str,
              input_shape: Tuple[int, int, int],
              output_shape: Tuple[int, int, int],
              data_filter=None,
              input_filter=None,
              output_filter=None) -> Tuple[np.ndarray, np.ndarray]:
    # Load image
    img = Image.open(filename)

    if data_filter is not None:
        # Apply filter to image
        img = np.array(img.resize(img.size))
        img = img.astype(float)
        img /= max(img.flat)
        img = data_filter(img) * 256
        img = img.astype(np.uint8)
        img = PIL.Image.fromarray(img)

    # Save filtered image as input data
    in_data = np.array(img.resize(input_shape[:2]), dtype=float)[:, :, :3]
    in_data /= max(in_data.flat)
    if input_filter is not None:
        in_data = input_filter(in_data)

    # Save raw image as output data
    out_data = np.array(img.resize(output_shape[:2]), dtype=float)[:, :, :3]
    out_data /= max(out_data.flat)
    if output_filter is not None:
        out_data = output_filter(out_data)

    return in_data, out_data


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
        in_data[i], out_data[i] = load_file(filename, input_shape, output_shape,
                                            data_filter=data_filter, input_filter=input_filter,
                                            output_filter=output_filter)
        print(f"\rLoaded {i + 1}/{len(data)}           ", end="")
    return in_data, out_data
