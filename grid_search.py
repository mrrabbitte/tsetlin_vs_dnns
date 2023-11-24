import json
import uuid
import datetime
import numpy as np

from experiment.experiment import get_experiments, train_test_split, run_experiment


def ensure_all_classes(x, y):
    x_train, y_train, x_test, y_test = train_test_split(x, y)

    while set(np.unique(y_train)) != set(np.unique(y_test)):
        x_train, y_train, x_test, y_test = train_test_split(x, y)

    return x_train, y_train, x_test, y_test


def grid_search(tm_grid, dnn_grid, run_for):
    experiments = get_experiments()

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

        x_train, y_train, x_test, y_test = ensure_all_classes(x, y)

        with open("grid-{0}-{1}.json".format(started_at, dataset_name), "w") as f:
            for (num_clauses, T, s, epochs) in tm_grid:
                params = {"num_clauses": int(num_clauses),
                          "T": T,
                          "s": s,
                          "epochs": int(epochs)}

                tms_result = run_experiment(
                    "tm", dataset_name,
                    x_train, y_train, x_test, y_test, preprocess_tm, train_tm, params)

                f.write(json.dumps({
                    "tm": tms_result.__dict__,
                    "tm_params": params,
                    "run_id": run_id
                }) + "\n")

                print(tms_result, params)

            for (batch_size, hidden_units, dropout, epochs) in dnn_grid:
                params = {"batch_size": int(batch_size),
                          "hidden_units": int(hidden_units),
                          "dropout": dropout,
                          "epochs": int(epochs)}

                dnns_result = run_experiment(
                    "dnn", dataset_name,
                    x_train, y_train, x_test, y_test, preprocess_dnn, train_dnn, params)

                f.write(json.dumps({
                    "dnn": dnns_result.__dict__,
                    "dnn_params": params,
                    "run_id": run_id
                }) + "\n")

                print(dnns_result, params)


if __name__ == "__main__":
    # TM params
    num_clausez = [50, 100, 150, 200, 250, 300]
    Tz = [0.01, 0.1, 1., 5., 10.]
    sz = [0.01, 0.1, 1.,  5., 10.]
    epochz = [10, 40, 80, 100, 200]

    tm_gridz = np.array(np.meshgrid(
        num_clausez,
        Tz,
        sz,
        epochz
    )).T.reshape(-1, 4)

    # DNN params
    batch_sizez = [10, 20, 60, 100, 150]
    hidden_unitz = [50, 100, 150, 250]
    dropoutz = [0.00000001, 0.00001, 0.1, 0.6, 0.9, 1.]
    epochz = [50, 100, 200]

    dnn_gridz = np.array(np.meshgrid(
        batch_sizez,
        hidden_unitz,
        dropoutz,
        epochz
    )).T.reshape(-1, 4)

    # Running grid search
    grid_search(tm_gridz, dnn_gridz, ["FLAGS", "GLASS", "SOYBEANS"])
