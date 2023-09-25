import json

from datasets.datasets import load_mnist
from training import hvr, mnist, sonar, wine
import numpy as np


def train_test_split(x, y, p_train=0.8):
    assert x.shape[0] == y.shape[0]

    num_train = int(p_train * x.shape[0])
    num_examples = x.shape[0]

    train_indices = np.random.choice(range(num_examples), size=(num_train,), replace=False)
    test_indices = np.array(list(set(range(num_examples)) - set(train_indices)))

    assert len(train_indices) == num_train
    assert len(set(train_indices).intersection(set(test_indices))) == 0
    assert len(set(train_indices).union(set(test_indices))) == num_examples

    return x[train_indices], y[train_indices], x[test_indices], y[test_indices]


def log(dataset, model, acc, took):
    logline = {"dataset": dataset, "model": model, "acc": acc, "took [ms]": took * 100}

    print(json.dumps(logline))


if __name__ == "__main__":
    num_bootstrap = 1

    x_mnist, y_mnist = load_mnist()

    experiments = {
        "MNIST": (x_mnist, y_mnist, mnist.run_tsetlin, mnist.run_dnn),
        "HVR": (x_mnist, y_mnist, hvr.run_tsetlin, hvr.run_dnn),
        "SONAR": (x_mnist, y_mnist, sonar.run_tsetlin, sonar.run_dnn),
        "WINE": (x_mnist, y_mnist, wine.run_tsetlin, wine.run_dnn)
    }

    for (dataset_name, experiment) in experiments.items():
        for i in range(num_bootstrap):
            (x, y, run_tsetlin, run_dnn) = experiment
            x_train, y_train, x_test, y_test = train_test_split(x, y)

            tsetlin_acc, tsetlin_time = run_tsetlin(x_train, y_train, x_test, y_test)
            log(dataset_name, "tsetlin", tsetlin_acc, tsetlin_time)

            dnn_acc, dnn_time = run_dnn(x_train, y_train, x_test, y_test)
            log(dataset_name, "dnn", dnn_acc, dnn_time)
