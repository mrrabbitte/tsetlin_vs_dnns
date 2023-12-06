
import numpy as np
import pandas as pd
import json

from scipy import spatial

from training import annealing, census, flags, soybeans, mnist, hvr
from datasets import datasets as data
from sklearn.feature_selection import mutual_info_classif


def identity(x):
    return x


class DatasetCharacteristic:

    def __init__(self,
                 dataset_name,
                 num_classes,
                 num_instances,
                 num_features,
                 class_balance,
                 mutal_info_median,
                 features_type):
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.num_features = num_features
        self.class_balance = class_balance
        self.mutal_info_median = mutal_info_median
        self.features_type = features_type

    def __str__(self):
        return json.dumps(self.__dict__)


def characterize(dataset_name,
                 loader,
                 features_type,
                 preprocess_X):
    X, y = loader()

    for i in range(0, len(y)):
        y[i] = str(y[i])

    X = preprocess_X(X)

    num_classes = len(np.unique(y))
    num_instances = np.shape(X)[0]
    num_features = np.shape(X)[1]
    mutual_info = mutual_info_classif(X, y)
    class_counts = pd.DataFrame(data=y, index=y).groupby(level=0).count().to_dict()[0]

    counts = list(class_counts.values())
    N = len(counts)
    cos_dis = spatial.distance.cosine(counts, np.ones(N) * (1. / N))

    return DatasetCharacteristic(dataset_name=dataset_name,
                                 num_classes=num_classes,
                                 num_instances=num_instances,
                                 num_features=num_features,
                                 mutal_info_median=np.median(mutual_info),
                                 class_balance=cos_dis,
                                 features_type=features_type)


if __name__ == "__main__":
    datasets = [
        ('BC', data.load_bc, 'numeric', identity),
        ('CENSUS', data.load_census, 'mixed', census.categorical_to_int),
        ('SONAR', data.load_sonar, 'numeric', identity),
        ('ANNEALING', data.load_annealing, 'mixed', annealing.categorical_to_int),
        ('GLASS', data.load_glass, 'numeric', identity),
        ('FLAGS', data.load_flags, 'mixed', flags.categorical_to_int),
        ('TUANDROMD', data.load_tuandromd, 'binary', identity),
        ('SOYBEANS', data.load_soybeans, 'mixed', soybeans.categorical_to_int),
        ('HVR', data.load_hvr, 'binary', hvr.categorical_to_int),
        ('MNIST', data.load_mnist, 'numeric', mnist.unroll)
    ]

    with open('characteristics.json', 'w') as f:
        for (ds_name, load, feat_type, pre_X) in datasets:
            result = characterize(ds_name, load, feat_type, pre_X)
            print(result)

            f.write(json.dumps(result.__dict__) + "\n")
