
import numpy as np

def run_tsetlin(x_train, y_train, x_test, y_test):
    x_train = np.where(x_train.reshape((x_train.shape[0], 28 * 28)) == 'y', 1, 0)
    x_test = np.where(x_test.reshape((x_test.shape[0], 28 * 28)) == 'y', 1, 0)

    

    return None

def run_dnn(x_train, y_train, x_test, y_test):
    return None