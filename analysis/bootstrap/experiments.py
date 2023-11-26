import json
import pandas as pd
import os

import pylab as py
import numpy as np

from scipy import spatial

def merge(res, metrics_name, params_name):
    merged = {}
    merged.update(res[metrics_name])
    merged.update(res[params_name])
    return merged


def read_characteristics():
    by_dataset = {}
    with open('characteristics.results', 'r') as f:
        for ln in f:
            characteristic = json.loads(ln)
            by_dataset.update({characteristic['dataset_name']: characteristic})
    return by_dataset


def compute_diffs(ds_res, name):
    vals = ds_res['tm_' + name] - ds_res['dnn_' + name]
    return np.mean(vals), np.std(vals)


def balance_measure(class_counts):
    consines = []

    print("CLASS COUNTS:", class_counts)

    for row in class_counts:
        print("CLASS COUNTS: ", row)
        counts = list(row.values())
        N = len(counts)
        consines.append(spatial.distance.cosine(counts, np.ones(N) * (1./N)))

    return consines


def mutual_info_measure(mutual_infos):
    mean_mutual_info = []
    for row in mutual_infos:
        mean_mutual_info.append(np.mean(row))
    return mean_mutual_info


def analyse(results_df: pd.DataFrame):
    datasets = results_df.dataset_name.unique()

    characteristics = read_characteristics()
    metrics = ["acc", "f1_median", "training_took_ms", "prediction_took_ms",
               "training_energy_micro_jules", "prediction_energy_micro_jules"]

    ignore_datasets = set(characteristics.keys()) - set(datasets)

    for dataset in ignore_datasets:
        print("Ignoring: ", characteristics.pop(dataset))

    for dataset in datasets:
        ds_res = results_df[results_df['dataset_name'] == dataset]

        data = characteristics[dataset]
        for metric in metrics:
            mean_diff, std_diff = compute_diffs(ds_res, metric)
            data.update({metric + "_diff_mean": mean_diff})
            data.update({metric + "_diff_std": std_diff})

    results = pd.DataFrame.from_records(list(characteristics.values()))

    results.to_csv('differences.csv')

    dataset_chars = [("num_classes", None),
                     ("num_instances", None),
                     ("num_features", None),
                     ("mutual_info", mutual_info_measure),
                     ("class_counts", balance_measure),
                     ("features_type", None)]
    show = True

    for (dataset_metric, processor) in dataset_chars:
        for i in range(0, len(metrics)):
            metric = metrics[i]

            print(dataset_metric, metric)

            x = results[dataset_metric]

            if processor is not None:
                print("Processing")
                x = processor(x)

            y = results[metric + "_diff_mean"]
            err = results[metric + "_diff_std"]

            print(x, y)
            
            py.title(dataset_metric + " vs. " + metric)
            py.scatter(x, y)
            py.errorbar(x, y, yerr=err, fmt='o')

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
        if result_file.startswith('characteristic'):
            continue

        with open(result_file) as f:
            print("Reading: " + result_file)
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

    analyse(results)
