import json
import pandas as pd
import os

import pylab as py
import numpy as np


def merge(res, metrics_name, params_name):
    merged = {}
    merged.update(res[metrics_name])
    merged.update(res[params_name])
    return merged


def plot_results(results_df: pd.DataFrame, tm=False):
    datasets = results_df.dataset_name.unique()

    for dataset in datasets:
        ds_res = results_df[results_df['dataset_name'] == dataset]

        y_name = 'batch_size'
        if tm:
            y_name = 'num_clauses'
        py.scatter(ds_res[y_name], ds_res['f1_median'])
        py.title("{0}: {1} - {2}".format(("TM" if tm else "DNN"), dataset, y_name), loc='center')
        py.show()

        counts, bins = np.histogram(ds_res['f1_median'])

        py.stairs(counts, bins)
        py.title(dataset, loc='center')
        py.show()


def get_winners(results_df: pd.DataFrame):
    datasets = results_df.dataset_name.unique()

    winners_per_dataset = {}

    for dataset in datasets:
        ds_res = results_df[results_df['dataset_name'] == dataset]
        ds_res = ds_res.sort_values(
            by=['f1_median', 'acc', 'training_energy_micro_jules'],
            ascending=[False, False, True])

        winners_per_dataset.update({dataset: ds_res.iloc[0]})

    return winners_per_dataset


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    dnn_results = []
    tm_results = []

    result_files = list(filter(lambda x: x.endswith('.results'), next(os.walk(os.getcwd()), (None, None, []))[2]))

    for result_file in result_files:
        with open(result_file) as f:
            for ln in f:
                result = json.loads(ln)

                if 'dnn' in result.keys():
                    dnn_results.append(merge(result, 'dnn', 'dnn_params'))
                elif 'tm' in result.keys():
                    tm_results.append(merge(result, 'tm', 'tm_params'))

    tm = pd.DataFrame.from_records(tm_results)
    dnn = pd.DataFrame.from_records(dnn_results)

    print(get_winners(tm))
    print(get_winners(dnn))

    plot_results(tm, True)
    plot_results(dnn)