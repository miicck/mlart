import numpy as np
from ml.load_data import data_directory, load_file, load_data
import ml
import os
import matplotlib.pyplot as plt

IN_SHAPE = (8, 8, 3)
OUT_SHAPE = (128, 128, 3)


def test_data_dir():
    assert os.path.isdir(data_directory())


def test_load_monochrome_file():
    inp, out = load_file(f"{data_directory()}/large/engraving/161.jpg", IN_SHAPE, OUT_SHAPE)
    assert inp.shape[:2] == (8, 8)
    assert out.shape[:2] == (128, 128)


def test_load_broken_file():
    inp, out = load_file(f"{data_directory()}/large/sculpture/374.jpg", IN_SHAPE, OUT_SHAPE)
    assert inp.shape[:2] == (8, 8)
    assert out.shape[:2] == (128, 128)
    assert np.allclose(inp, 0)
    assert np.allclose(out, 0)


def test_load():
    inp, out = load_file(f"{data_directory()}/large/painting/1300.jpg", IN_SHAPE, OUT_SHAPE)
    assert len(inp.shape) == 3
    assert len(out.shape) == 3


def test_load_2():
    inp, out = load_file(f"{data_directory()}/large/sculpture/190 18.59.45.jpg", IN_SHAPE, OUT_SHAPE)
    assert len(inp.shape) == 3
    assert len(out.shape) == 3


def test_load_3():
    in_data = np.zeros((1, 8, 8, 3))
    out_data = np.zeros((1, 128, 128, 3))
    result = load_file(f"{data_directory()}/large/painting/1225.jpg", IN_SHAPE, OUT_SHAPE)
    in_data[0] = result[0]
    out_data[0] = result[1]
    in_data[0], out_data[0] = result


def test_load_dir():
    load_data(f"{data_directory()}/large/painting", IN_SHAPE, OUT_SHAPE, max_count=100, shuffle=False)
