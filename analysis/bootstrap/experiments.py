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


def plot_results(results_df: pd.DataFrame):
    datasets = results_df.dataset_name.unique()

    for dataset in datasets:
        ds_res = results_df[results_df['dataset_name'] == dataset]

        py.subplots(3, 1)

        py.subplot(311)
        vals = ds_res['tm_acc']
        counts, bins = np.histogram(vals)
        py.stairs(counts, bins)
        py.title("TM - " + dataset + " - Mean: {0}, Std: {1}".format(np.mean(vals), np.std(vals)),
                 loc='center')

        py.subplot(312)
        vals = ds_res['dnn_acc']
        counts, bins = np.histogram(vals)
        py.stairs(counts, bins)
        py.title("DNN - " + dataset + " - Mean: {0}, Std: {1}".format(np.mean(vals), np.std(vals)), loc='center')

        py.subplot(313)
        vals = ds_res['tm_acc'] - ds_res['dnn_acc']
        counts, bins = np.histogram(vals)
        py.stairs(counts, bins)
        py.title("Difference - " + dataset + " - Mean: {0}, Std: {1}".format(np.mean(vals), np.std(vals)), loc='center')

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


def prefix(record, name):
    updated = {}
    for key in record.keys():
        if key == 'dataset_name':
            updated.update({key: record[key]})
            continue
        updated.update({name + "_" + key: record[key]})
    return updated


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    results = []

    result_files = list(filter(lambda x: x.endswith('.results'), next(os.walk(os.getcwd()), (None, None, []))[2]))

    for result_file in result_files:
        with open(result_file) as f:
            for ln in f:
                result = json.loads(ln)

                dnn = result.pop('dnn')
                tm = result.pop('tm')

                merged = {}
                merged.update(prefix(dnn, 'dnn'))
                merged.update(prefix(tm, 'tm'))
                merged.update(result)

                results.append(merged)

    results = pd.DataFrame.from_records(results)

    plot_results(results)
