from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical

from sklearn.preprocessing import OneHotEncoder

CATEGORIES = [
    ['0', '1', '2', '3', '4', '5', '6', '?'],
    ['0', '1', '?'],
    ['0', '1', '2', '?'],
    ['0', '1', '2', '?'],
    ['0', '1', '?'],
    ['0', '1', '2', '3', '?'],
    ['0', '1', '2', '3', '?'],
    ['0', '1', '2', '?'],
    ['0', '1', '2', '?'],
    ['0', '1', '2', '?'],
    ['0', '1', '?'],
    [0, 1],
    ['0', '1', '2', '?'],
    ['0', '1', '2', '?'],
    ['0', '1', '2', '?'],
    ['0', '1', '?'],
    ['0', '1', '?'],
    ['0', '1', '2', '?'],
    ['0', '1', '?'],
    ['0', '1', '?'],
    ['0', '1', '2', '3', '?'],
    ['0', '1', '2', '3', '?'],
    ['0', '1', '?'],
    ['0', '1', '?'],
    ['0', '1', '?'],
    ['0', '1', '2', '?'],
    ['0', '1', '?'],
    ['0', '1', '2', '3', '?'],
    ['0', '1', '2', '4', '?'],
    ['0', '1', '?'],
    ['0', '1', '?'],
    ['0', '1', '?'],
    ['0', '1', '?'],
    ['0', '1', '?'],
    ['0', '1', '2', '?']
]


def preprocess_tsetlin(x, y):
    return x, __preprocess_y(y)


def train_tsetlin(x, y, num_clauses=250, T=10., s=5., epochs=80):
    categorical_encoder = OneHotEncoder(categories=CATEGORIES)
    categorical_encoder.fit(x)

    encode_x = lambda data: categorical_encoder.transform(data).toarray()

    x = encode_x(x)

    tm = MultiClassTsetlinMachine(num_clauses, T, s)
    tm.fit(x, y, epochs=epochs, incremental=True)

    return lambda x_test: tm.predict(encode_x(x_test))


def preprocess_dnn(x, y):
    return x, to_categorical(__preprocess_y(y))


def train_dnn(x, y, batch_size=60, hidden_units=150, dropout=0.00000001, epochs=200):
    categorical_encoder = OneHotEncoder(categories=CATEGORIES)
    categorical_encoder.fit(x)

    encode_x = lambda data: categorical_encoder.transform(data).toarray().astype('float32')

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
    categories = ['2-4-d-injury', 'alternarialeaf-spot', 'anthracnose', 'bacterial-blight',
                  'bacterial-pustule', 'brown-spot', 'brown-stem-rot', 'charcoal-rot', 'cyst-nematode',
                  'diaporthe-pod-&-stem-blight', 'diaporthe-stem-canker', 'downy-mildew', 'frog-eye-leaf-spot',
                  'herbicide-injury', 'phyllosticta-leaf-spot', 'phytophthora-rot', 'powdery-mildew',
                  'purple-seed-stain', 'rhizoctonia-root-rot']

    return np.vectorize(lambda cat: categories.index(cat))(y)


def replace(col, catz):
    for i in range(0, len(col)):
        col[i] = catz.index(col[i])
    return col


def categorical_to_int(X):
    for i in range(0, len(CATEGORIES)):
        col_num = i
        X[:, col_num] = replace(X[:, col_num], CATEGORIES[i])
    return X