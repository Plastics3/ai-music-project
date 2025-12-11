"""Helper utilities for the project."""
from PIL import Image
import numpy as np


def load_image_as_tensor(path_or_file):
    img = Image.open(path_or_file).convert('L').resize((28,28))
    arr = np.array(img).astype('float32') / 255.0
    arr = (arr - 0.1307) / 0.3081
    arr = arr[np.newaxis, np.newaxis, :, :]
    return arr
