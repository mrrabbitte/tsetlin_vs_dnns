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
