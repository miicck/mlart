import numpy as np
from ml.load_data import data_directory, load_file
import ml
import os
import matplotlib.pyplot as plt


def test_data_dir():
    assert os.path.isdir(data_directory())


def test_load_monochrome_file():
    inp, out = load_file(f"{data_directory()}/large/engraving/161.jpg", (8, 8), (128, 128))
    assert inp.shape[:2] == (8, 8)
    assert out.shape[:2] == (128, 128)


def test_load_broken_file():
    inp, out = load_file(f"{data_directory()}/large/sculpture/374.jpg", (8, 8), (128, 128))
    assert inp.shape[:2] == (8, 8)
    assert out.shape[:2] == (128, 128)
    assert np.allclose(inp, 0)
    assert np.allclose(out, 0)


def test_load():
    inp, out = load_file(f"{data_directory()}/large/painting/1300.jpg", (8, 8), (128, 128))
    assert len(inp.shape) == 3
    assert len(out.shape) == 3


def test_load_2():
    inp, out = load_file(f"{data_directory()}/large/sculpture/190 18.59.45.jpg", (8, 8), (128, 128))
    assert len(inp.shape) == 3
    assert len(out.shape) == 3
