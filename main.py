import datetime

import uuid

import tensorflow

from experiment.experiment import train_test_split, run_experiment, get_experiments

import json


def main():
    # Specification
    experiments = get_experiments()

    # Setup
    gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tensorflow.config.experimental.set_memory_growth(device, True)

    # Config
    run_for = ["TUANDROMD"]
    n_bootstrap = 1

    # Execution
    started_at = datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
    run_id = str(uuid.uuid4())

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

