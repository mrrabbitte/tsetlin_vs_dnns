
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine

from time import time


def run_tsetlin(x_train, y_train, x_test, y_test):
    tm = MultiClassTsetlinMachine(100, 10.0, 2.0)

    start_training = time()
    tm.fit(x_train, y_train, epochs=10, incremental=True)
    stop_training = time()

    acc = 100 * (tm.predict(x_test) == y_test).mean()

    return acc, stop_training - start_training


def run_dnn(x_train, y_train, x_test, y_test):
    return 1, 1