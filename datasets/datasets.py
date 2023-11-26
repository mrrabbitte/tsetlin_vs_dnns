import numpy as np
import pandas as pd

from keras.datasets import mnist

from pkg_resources import resource_stream


# @article{deng2012mnist,
#   title={The mnist database of handwritten digit images for machine learning research},
#   author={Deng, Li},
#   journal={IEEE Signal Processing Magazine},
#   volume={29},
#   number={6},
#   pages={141--142},
#   year={2012},
#   publisher={IEEE}
# }
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    return x, y


# @misc{misc_congressional_voting_records_105,
#   title        = {{Congressional Voting Records}},
#   year         = {1987},
#   howpublished = {UCI Machine Learning Repository},
#   note         = {{DOI}: https://doi.org/10.24432/C5C01P}
# }
def load_hvr():
    arr = pd.read_csv(resource_stream("datasets.data.hvr", "house-votes-84.data")).to_numpy()

    x = arr[:, 1:]
    y = arr[:, 0]

    return x, y


# @misc{misc_connectionist_bench_(sonar,_mines_vs._rocks)_151,
#   author       = {Sejnowski,Terry and Gorman,R.},
#   title        = {{Connectionist Bench (Sonar, Mines vs. Rocks)}},
#   howpublished = {UCI Machine Learning Repository},
#   note         = {{DOI}: https://doi.org/10.24432/C5T01Q}
# }
def load_sonar():
    arr = pd.read_csv(resource_stream("datasets.data.sonar", "sonar_csv.csv")).to_numpy()

    x = arr[:, :60]
    y = arr[:, 60]

    return x, y


# @misc{misc_breast_cancer_wisconsin_(diagnostic)_17,
#   author       = {Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.},
#   title        = {{Breast Cancer Wisconsin (Diagnostic)}},
#   year         = {1995},
#   howpublished = {UCI Machine Learning Repository},
#   note         = {{DOI}: https://doi.org/10.24432/C5DW2B}
# }
def load_bc():
    arr = pd.read_csv(resource_stream("datasets.data.bc", "wdbc.data")).to_numpy()

    x = arr[:, 2:]
    y = arr[:, 1]

    return x, y


# @misc{misc_tuandromd_(tezpur_university_android_malware_dataset)_855,
#   author       = {Borah,Parthajit and Bhattacharyya,Dhruba K.},
#   title        = {{TUANDROMD (Tezpur University Android Malware Dataset)}},
#   year         = {2023},
#   howpublished = {UCI Machine Learning Repository},
#   note         = {{DOI}: https://doi.org/10.24432/C5560H}
# }
def load_tuandromd():
    arr = pd.read_csv(resource_stream("datasets.data.tuandromd", "TUANDROMD.csv")).to_numpy()

    x = arr[:, :241]
    y = arr[:, 241]

    return x, y


# @misc{misc_adult_2,
#   author       = {Becker,Barry and Kohavi,Ronny},
#   title        = {{Adult}},
#   year         = {1996},
#   howpublished = {UCI Machine Learning Repository},
#   note         = {{DOI}: https://doi.org/10.24432/C5XW20}
# }
def load_census():
    data = pd.read_csv(resource_stream("datasets.data.census", "adult.data"))
    test = pd.read_csv(resource_stream("datasets.data.census", "adult.test"))

    data = strip_strings(data).to_numpy()
    test = strip_strings(test).to_numpy()

    x = np.concatenate((data[:, :14], test[:, :14]))
    y = np.concatenate((data[:, 14], test[:, 14]))

    return x, y


# @misc{misc_annealing_3,
#   title        = {{Annealing}},
#   howpublished = {UCI Machine Learning Repository},
#   note         = {{DOI}: https://doi.org/10.24432/C5RW2F}
# }
# Classes: 1,2,3,4,5,U
def load_annealing():
    data = pd.read_csv(resource_stream("datasets.data.annealing", "anneal.data"))
    test = pd.read_csv(resource_stream("datasets.data.annealing", "anneal.test"))

    data = strip_strings(data).to_numpy()
    test = strip_strings(test).to_numpy()

    x = np.concatenate((data[:, :38], test[:, :38]))
    y = np.concatenate((data[:, 38], test[:, 38]))

    return x, y


# @misc{misc_flags_40,
#   title        = {{Flags}},
#   year         = {1990},
#   howpublished = {UCI Machine Learning Repository},
#   note         = {{DOI}: https://doi.org/10.24432/C52C7Z}
# }
def load_flags():
    data = pd.read_csv(resource_stream("datasets.data.flags", "flag.data"))

    data = strip_strings(data).to_numpy()

    x = np.concatenate((data[:, 1:6], data[:, 7:]), axis=1)
    y = data[:, 6]  # Taking religion as the category

    return x, y


# @misc{misc_soybean_(large)_90,
#   author       = {Michalski,R.S. and Chilausky,R.L.},
#   title        = {{Soybean (Large)}},
#   year         = {1988},
#   howpublished = {UCI Machine Learning Repository},
#   note         = {{DOI}: https://doi.org/10.24432/C5JG6Z}
# }
def load_soybeans():
    data = pd.read_csv(resource_stream("datasets.data.soybeans", "soybean-large.data"))

    data = strip_strings(data).to_numpy()

    x = data[:, 1:]
    y = data[:, 0]

    return x, y


# @misc{misc_glass_identification_42,
#   author       = {German,B.},
#   title        = {{Glass Identification}},
#   year         = {1987},
#   howpublished = {UCI Machine Learning Repository},
#   note         = {{DOI}: https://doi.org/10.24432/C5WW2P}
# }
def load_glass():
    data = pd.read_csv(resource_stream("datasets.data.glass", "glass.data")).to_numpy()

    x = data[:, 1:10]
    y = data[:, 10]

    return x, y


# This is to sanitize the data and get rid of trivial bugs downstream.
def strip_strings(df):
    strings = df.select_dtypes(['object'])
    df[strings.columns] = strings.apply(lambda x: x.str.strip())
    return df
