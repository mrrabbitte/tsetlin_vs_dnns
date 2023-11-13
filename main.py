import datetime

import uuid

import tensorflow

from datasets.datasets import load_mnist, load_hvr, load_bc, load_sonar, load_tuandromd, load_census, load_annealing
from measurement import power
from training import hvr, mnist, sonar, bc, tuandromd, census, annealing
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import json


class ExperimentResult:

    def __init__(self, model_name, dataset_name, training_took_ms, prediction_took_ms, f1_scores, f1_median, acc,
                 training_energy_micro_jules, prediction_energy_micro_jules):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.training_took_ms = training_took_ms
        self.prediction_took_ms = prediction_took_ms
        self.f1_scores = list(f1_scores)
        self.f1_median = f1_median
        self.acc = acc
        self.training_energy_micro_jules = training_energy_micro_jules
        self.prediction_energy_micro_jules = prediction_energy_micro_jules

    def __str__(self):
        return json.dumps(self.__dict__)


def time_ms():
    return time() * 1000


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


def scores_to_one_hot(scores):
    num_scores = np.shape(scores)[0]
    classes = np.zeros(np.shape(scores))

    for i in range(0, num_scores):
        score_row = scores[i]

        max_score = np.max(score_row)
        n_classes = len(score_row)

        for j in range(0, n_classes):
            if score_row[j] == max_score:
                classes[i, j] = 1
            else:
                classes[i, j] = 0
    return classes


def one_hot_to_classes(scores):
    num_scores = np.shape(scores)[0]
    classes = np.zeros((num_scores, ))
    for i in range(0, num_scores):
        score = scores[i]
        for clazz in range(0, len(score)):
            if score[clazz] == 1:
                classes[i] = clazz
    return classes


def run_experiment(model_name, dataset_name, x_train, y_train, x_test, y_test, preprocess, train):
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)

    power_training_at = power.start_power()
    started_training_at = time_ms()
    predict = train(x_train, y_train)
    training_took_ms = time_ms() - started_training_at
    trainig_took_micro_jules = power.get_power_sum(power_training_at, power.stop_power())

    power_predict_at = power.start_power()
    started_predict_at = time_ms()
    y_pred = predict(x_test)
    prediction_took_ms = time_ms() - started_predict_at
    prediction_took_micro_jules = power.get_power_sum(power_predict_at, power.stop_power())

    if model_name == "dnn":
        y_test, y_pred = one_hot_to_classes(y_test), one_hot_to_classes(scores_to_one_hot(y_pred))

    f1_scores = f1_score(y_test, y_pred, average=None)
    acc = accuracy_score(y_test, y_pred)

    return ExperimentResult(
        model_name,
        dataset_name,
        training_took_ms,
        prediction_took_ms,
        f1_scores,
        np.median(f1_scores),
        acc,
        trainig_took_micro_jules,
        prediction_took_micro_jules)


def main():
    # Specification
    experiments = {
        "MNIST": (load_mnist, mnist.preprocess_tsetlin, mnist.train_tsetlin,
                      mnist.preprocess_dnn, mnist.train_dnn),
        "HVR": (load_hvr, hvr.preprocess_tsetlin, hvr.train_tsetlin, hvr.preprocess_dnn, hvr.train_dnn),
        "BC": (load_bc, bc.preprocess_tsetlin, bc.train_tsetlin, bc.preprocess_dnn, bc.train_dnn),
        "SONAR": (load_sonar, sonar.preprocess_tsetlin, sonar.train_tsetlin, sonar.preprocess_dnn, sonar.train_dnn),
        "TUANDROMD": (load_tuandromd,
                      tuandromd.preprocess_tsetlin, tuandromd.train_tsetlin,
                      tuandromd.preprocess_dnn, tuandromd.train_dnn),
        "CENSUS": (load_census,
                   census.preprocess_tsetlin, census.train_tsetlin, census.preprocess_dnn, census.train_dnn),
        "ANNEALING": (load_annealing, annealing.preprocess_tsetlin, annealing.train_tsetlin,
                      annealing.preprocess_dnn, annealing.train_dnn)
    }

    # Setup
    gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tensorflow.config.experimental.set_memory_growth(device, True)

    # Config
    run_for = ["ANNEALING"]
    n_bootstrap = 1

    started_at = datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
    run_id = str(uuid.uuid4())

    # Execution
    for (dataset_name, experiment) in experiments.items():
        if dataset_name not in run_for:
            continue

        print("Running experiment for dataset: {0} with run id: {1}, started at: {2}".format(
            dataset_name, run_id, started_at))

        (loader, preprocess_tm, train_tm, preprocess_dnn, train_dnn) = experiment

        x, y = loader()

        with open("{0}-{1}.json".format(started_at, dataset_name), "w") as f:
            for i in range(n_bootstrap):
                x_train, y_train, x_test, y_test = train_test_split(x, y)

                tms_result = run_experiment(
                     "tm", dataset_name, x_train, y_train, x_test, y_test, preprocess_tm, train_tm)

                print(tms_result)

                dnns_result = run_experiment(
                    "dnn", dataset_name, x_train, y_train, x_test, y_test, preprocess_dnn, train_dnn)

                print(dnns_result)

                f.write(json.dumps({
                    "tm": tms_result.__dict__,
                    "dnn": dnns_result.__dict__,
                    "run_id": run_id,
                    "boots_i": i,
                    "N_boots": n_bootstrap
                }) + "\n")


if __name__ == "__main__":
    main()

