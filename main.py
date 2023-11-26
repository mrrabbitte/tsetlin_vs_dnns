import datetime

import uuid

import tensorflow

from experiment.experiment import train_test_split, run_experiment, get_experiments

import json
import numpy as np


def discarded(x, val):
    x.discard(val)
    return x


def ensure_all_classes(x, y):
    x_train, y_train, x_test, y_test = train_test_split(x, y)

    while (discarded(set(np.unique(y_train)), '2-4-d-injury')
           != discarded(set(np.unique(y_test)), '2-4-d-injury')):
        x_train, y_train, x_test, y_test = train_test_split(x, y)

    return x_train, y_train, x_test, y_test

def main():
    # Specification
    experiments = get_experiments()

    # Setup
    gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tensorflow.config.experimental.set_memory_growth(device, True)

    # Config
    run_for = ["CENSUS"]
    n_bootstrap = 250

    # Execution
    started_at = datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
    run_id = str(uuid.uuid4())

    total = 1. * len(run_for) * n_bootstrap
    num_runs = 0.

    for (dataset_name, experiment) in experiments.items():
        if dataset_name not in run_for:
            continue

        print("Running experiment for dataset: {0} with run id: {1}, started at: {2}".format(
            dataset_name, run_id, started_at))

        (loader, preprocess_tm, train_tm, preprocess_dnn, train_dnn) = experiment

        x, y = loader()

        with open("boots-{0}-{1}.json".format(started_at, dataset_name), "w") as f:
            for i in range(n_bootstrap):
                x_train, y_train, x_test, y_test = ensure_all_classes(x, y)

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

                num_runs += 1.
                print("Progress: ", num_runs/total)


if __name__ == "__main__":
    main()

