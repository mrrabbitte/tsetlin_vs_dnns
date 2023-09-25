import numpy as np
import pandas as pd

from keras.datasets import mnist


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    return x, y


def load_hvr(hvr_path):
    df = pd.read_csv(hvr_path)

    arr = df.to_numpy()

    x = arr[:, 1:]
    y = arr[:, 0]

    return x, y


if __name__ == "__main__":
    load_hvr("/home/mrrabbit/Data/datasets/hvr/house-votes-84.data")