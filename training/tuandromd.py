from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine

from time import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical

import numpy as np


def preprocess_tsetlin(x, y):
    return x, __preprocess_y(y)


def train_tsetlin(x_train, y_train):
    tm = MultiClassTsetlinMachine(100, 5, 1)

    tm.fit(x_train, y_train, epochs=10, incremental=True)

    return lambda x_test: tm.predict(x_test)


def preprocess_dnn(x, y):
    return x.astype('float32'), to_categorical(__preprocess_y(y))


def train_dnn(x_train, y_train):
    num_labels = len(np.unique(y_train))

    batch_size = 20
    hidden_units = 200
    dropout = 0.3

    model = Sequential()
    model.add(Dense(hidden_units, input_dim=x_train.shape[1]))
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

    model.fit(x_train, y_train, epochs=120, batch_size=batch_size, verbose=0)

    return lambda x_test: model.predict(x_test, batch_size=batch_size)


def __preprocess_y(y):
    return np.where(y == 'malware', 1, 0)
