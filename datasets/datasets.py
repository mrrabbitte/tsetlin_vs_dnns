import numpy as np
from keras.datasets import mnist


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
