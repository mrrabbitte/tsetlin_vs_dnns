import numpy as np
import pandas as pd

from keras.datasets import mnist


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    return x, y


def load_hvr(hvr_path):
    arr = pd.read_csv(hvr_path).to_numpy()

    x = arr[:, 1:]
    y = arr[:, 0]

    return x, y


def load_sonar(sonar_path):
    arr = pd.read_csv(sonar_path).to_numpy()

    x = arr[:, :60]
    y = arr[:, 60:]

    return x, y


def load_bc(bc_path):
    arr = pd.read_csv(bc_path).to_numpy()

    x = arr[:, 2:]
    y = arr[:, 1]

    return x, y


if __name__ == "__main__":
    load_hvr("/home/mrrabbit/Data/datasets/hvr/house-votes-84.data")
    load_sonar("/home/mrrabbit/Data/datasets/sonar/sonar_csv.csv")
    load_bc("/home/mrrabbit/Data/datasets/bc/wdbc.data")
