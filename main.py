import json

from datasets.datasets import load_mnist, load_hvr, load_bc, load_sonar, load_tuandromd, load_census
from training import hvr, mnist, sonar, bc, tuandromd, census
import numpy as np


def train_test_split(X, y, p_train=0.8):
    assert X.shape[0] == y.shape[0]

    num_train = int(p_train * X.shape[0])
    num_examples = X.shape[0]

    train_indices = np.random.choice(range(num_examples), size=(num_train,), replace=False)
    test_indices = np.array(list(set(range(num_examples)) - set(train_indices)))

    assert len(train_indices) == num_train
    assert len(set(train_indices).intersection(set(test_indices))) == 0
    assert len(set(train_indices).union(set(test_indices))) == num_examples

    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def log(dataset, model, acc, took):
    logline = {"dataset": dataset, "model": model, "acc": acc, "took [s]": took}

    print(json.dumps(logline))


if __name__ == "__main__":
    # Specification
    experiments = {
        "MNIST": (load_mnist, mnist.run_tsetlin, mnist.run_dnn),
        "HVR": (load_hvr, hvr.run_tsetlin, hvr.run_dnn),
        "BC": (load_bc, bc.run_tsetlin, bc.run_dnn),
        "SONAR": (load_sonar, sonar.run_tsetlin, sonar.run_dnn),
        "TUANDROMD": (load_tuandromd, tuandromd.run_tsetlin, tuandromd.run_dnn),
        "CENSUS": (load_census, census.run_tsetlin, census.run_dnn)
    }

    # Config
    run_for = ["CENSUS"]
    num_bootstrap = 1

    # Execution
    for (dataset_name, experiment) in experiments.items():
        if dataset_name not in run_for:
            continue

        print("Running experiment for dataset: ", dataset_name)

        (loader, run_tsetlin, run_dnn) = experiment

        x, y = loader()

        for i in range(num_bootstrap):

            x_train, y_train, x_test, y_test = train_test_split(x, y)

            tsetlin_acc, tsetlin_time = run_tsetlin(x_train, y_train, x_test, y_test)
            log(dataset_name, "tsetlin", tsetlin_acc, tsetlin_time)

            dnn_acc, dnn_time = run_dnn(x_train, y_train, x_test, y_test)
            log(dataset_name, "dnn", dnn_acc, dnn_time)
