import json
import pandas as pd
import os

import pylab as py
import numpy as np

from scipy import spatial

import scipy.stats as st

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


def plot_hist(title, vals):
    counts, bins = np.histogram(vals, bins=100)
    py.title(title + ", Normal-test:" + str(st.normaltest(vals)))
    py.stairs(counts, bins, fill=True)
    py.show()


def compute_diffs(dataset_name, ds_res, name):
    tm_vals = ds_res['tm_' + name]
    dnn_vals = ds_res['dnn_' + name]

    #plot_hist("TM " + name, tm_vals)
    #plot_hist("DNN " + name, dnn_vals)

    U_stat, p_diff = st.mannwhitneyu(tm_vals, dnn_vals)

    if p_diff < 0.0001:
        print("Significant difference: {0}, (p={1}), dataset: {2}, metric: {3} ".format(
            U_stat, p_diff, dataset_name, name))

    vals = tm_vals - dnn_vals

    #plot_hist("Dataset: {0}, metric: {1}".format(dataset_name, name), vals)
    return U_stat, p_diff, np.median(vals), np.std(vals)


def balance_measure(class_counts):
    consines = []

    for row in class_counts:
        counts = list(row.values())
        N = len(counts)
        cos_dis = spatial.distance.cosine(counts, np.ones(N) * (1./N))
        consines.append(cos_dis)
        print("CLASSES: {0}, cosine: {1}".format(row, cos_dis))

    return consines


def mutual_info_measure(mutual_infos):
    mean_mutual_info = []
    for row in mutual_infos:
        mean_mutual_info.append(np.mean(row))
    return mean_mutual_info


def analyse(results_df: pd.DataFrame):
    datasets = results_df.dataset_name.unique()

    characteristics = read_characteristics()
    metrics = [
        ("acc", "Accuracy", ""),
        ("f1_median", "F1 Median", ""),
        ("training_took_ms", "Training Time", "[ms]"),
        ("prediction_took_ms", "Prediction Time", "[ms]"),
        ("training_energy_micro_jules", "Training Energy", "[µJ]"),
        ("prediction_energy_micro_jules", "Prediction Energy", "[µJ]")]

    ignore_datasets = set(characteristics.keys()) - set(datasets)

    for dataset in ignore_datasets:
        print("Ignoring: ", characteristics.pop(dataset))

    for dataset in datasets:
        ds_res = results_df[results_df['dataset_name'] == dataset]

        data = characteristics[dataset]
        for (metric, _, _) in metrics:
            U_stat, p_val, mean_diff, std_diff = compute_diffs(dataset, ds_res, metric)

            data.update({metric + "_diff_median": mean_diff})
            data.update({metric + "_diff_std": std_diff})
            data.update({metric + "_diff_U_stat": U_stat})
            data.update({metric + "_diff_p_value": p_val})
            print("Dataset: {0}, metric: {1}, U: {2}, p-value: {3}".format(dataset, metric, U_stat, p_val))

    results = pd.DataFrame.from_records(list(characteristics.values()))

    results.to_csv('differences.csv')

    dataset_chars = [("num_classes", None, "Number of classes", "[#]"),
                     ("num_instances", None, "Number of instances", "[#]"),
                     ("num_features", None,  "Number of features", "[#]"),
                     ("mutual_info", mutual_info_measure, "Median of Mutual Information", "[bit]"),
                     ("class_counts", balance_measure, "Class Balance", "")]
    show = True

    for (dataset_metric, processor, ds_metric_nice_name, ds_metric_unit_maybe) in dataset_chars:
        for i in range(0, len(metrics)):
            (metric, metric_nice_name, metric_unit) = metrics[i]

            print(dataset_metric, metric)

            x = results[dataset_metric]

            if processor is not None:
                print("Processing")
                x = processor(x)

            y = results[metric + "_diff_median"]
            err = results[metric + "_diff_std"]

            rho, p = st.spearmanr(x, y)

            print(metric, rho, p)
            if p < 0.05:
                py.scatter(x, y)
                py.xlabel("{0} {1}".format(ds_metric_nice_name, ds_metric_unit_maybe))
                py.ylabel("{0} {1}".format(metric_nice_name, metric_unit))
                py.grid(visible=True)
            #py.errorbar(x, y, yerr=err, fmt='o')

            py.show()

    print(results)


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
