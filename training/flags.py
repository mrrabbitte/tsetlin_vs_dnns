from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder

CATEGORIES = [
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [0, 1, 2, 3, 5],
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 14],
    [1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    ['black', 'blue', 'brown', 'gold', 'green', 'orange', 'red', 'white'],
    [0, 1, 2, 4],
    [0, 1, 2],
    [0, 1],
    [0, 1, 4],
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 14, 15, 22, 50],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    ['black', 'blue', 'gold', 'green', 'orange', 'red', 'white'],
    ['black', 'blue', 'brown', 'gold', 'green', 'orange', 'red', 'white']
]


CATEGORICAL_COLUMNS = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
NUMERIC_COLUMNS = [2, 3]


def preprocess_tsetlin(x, y):
    return x, y


def train_tsetlin(x, y):
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

    tm = MultiClassTsetlinMachine(250, 10, 1)
    tm.fit(x, y, epochs=50, incremental=True)

    return lambda x_test: tm.predict(encode_x(x_test))


def preprocess_dnn(x, y):
    return x, to_categorical(y)


def train_dnn(x, y):
    categorical_encoder = OneHotEncoder(categories=CATEGORIES)
    categorical_encoder.fit(x[:, CATEGORICAL_COLUMNS])

    encode_x = lambda data: np.concatenate(
        (categorical_encoder.transform(data[:, CATEGORICAL_COLUMNS]).toarray(),
         data[:, NUMERIC_COLUMNS]), axis=1).astype('float32')
    x = encode_x(x)

    # compute the number of labels
    num_labels = np.shape(y)[1]

    # network parameters
    batch_size = 50
    hidden_units = 250
    dropout = 0.1

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

    model.fit(x, y, epochs=120, batch_size=batch_size, verbose=0)

    return lambda x_test: model.predict(encode_x(x_test), batch_size)

