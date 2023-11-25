from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder

CATEGORIES = [
    ["GB", "GK", "GS", "TN", "ZA", "ZF", "ZH", "ZM", "ZS", "?"],
    ["C", "H", "G"],
    ["R", "A", "U", "K", "M", "S", "W", "V", "?"],
    ["T", "?"],
    ["S", "A", "X", "?"],
    ["1", "2", "3", "4", "5", "?"],
    ["N", "?"],
    ["P", "M", "?"],
    ["D", "E", "F", "G", "?"],
    ["1", "2", "3", "4", "5", "?"],
    ["Y", "?"],
    ["Y", "?"],
    ["Y", "?"],
    ["B", "M", "?"],
    ["Y", "?"],
    ["Y", "?"],
    ["C", "?"],
    ["P", "?"],
    ["Y", "?"],
    ["Y", "?"],
    ["Y", "?"],
    ["Y", "?"],
    ["Y", "?"],
    ["B", "R", "V", "C", "?"],
    ["Y", "?"],
    ["Y", "?"],
    ["Y", "?"],
    ["Y", "?"],
    ["COIL", "SHEET", "?"],
    ["Y", "N", "?"],
    [0, 500, 600, 760],
    ["1", "2", "3", "?"]
]

CATEGORICAL_COLUMNS = [0, 1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                       29, 30, 31, 35, 36, 37]
NUMERIC_COLUMNS = [3, 4, 8, 32, 33, 34]


def preprocess_tsetlin(x, y):
    return x, __preprocess_y(y)


def train_tsetlin(x, y, num_clauses=100, T=5., s=10., epochs=80):
    categorical_encoder = OneHotEncoder(categories=CATEGORIES)
    categorical_encoder.fit(x[:, CATEGORICAL_COLUMNS])

    discretizer = KBinsDiscretizer(encode="onehot-dense",
                                   strategy="quantile",
                                   n_bins=3)
    discretizer.fit(x[:, NUMERIC_COLUMNS])

    encode_x = lambda data: np.concatenate(
        (categorical_encoder.transform(data[:, CATEGORICAL_COLUMNS]).toarray(),
         discretizer.transform(data[:, NUMERIC_COLUMNS])), axis=1)

    x = encode_x(x)

    tm = MultiClassTsetlinMachine(num_clauses, T, s)
    tm.fit(x, y, epochs=epochs, incremental=True)

    return lambda x_test: tm.predict(encode_x(x_test))


def preprocess_dnn(x, y):
    return x, to_categorical(__preprocess_y(y))


def train_dnn(x, y, batch_size=10, hidden_units=100, dropout=0.00001, epochs=200):
    categorical_encoder = OneHotEncoder(categories=CATEGORIES)
    categorical_encoder.fit(x[:, CATEGORICAL_COLUMNS])

    encode_x = lambda data: np.concatenate(
        (categorical_encoder.transform(data[:, CATEGORICAL_COLUMNS]).toarray(),
         data[:, NUMERIC_COLUMNS]), axis=1).astype('float32')
    x = encode_x(x)

    num_labels = np.shape(y)[1]

    model = Sequential()
    model.add(Dense(hidden_units, input_dim=x.shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_units))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

    return lambda x_test: model.predict(encode_x(x_test), batch_size)


def __preprocess_y(y):
    categories = {
        "1": 0,
        "2": 1,
        "3": 2,  # 4 has 0 instances, so we will ignore it.
        "5": 3,
        "U": 4
    }

    return np.vectorize(lambda cat: categories[cat])(y)
