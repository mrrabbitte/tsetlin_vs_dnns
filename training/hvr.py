from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine

from time import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical

import numpy as np


def run_tsetlin(x_train, y_train, x_test, y_test):
    x_train = __preprocess_x(x_train)
    x_test = __preprocess_x(x_test)

    y_train = __preprocess_y(y_train)
    y_test = __preprocess_y(y_test)

    print(x_train[0])
    print(y_train[0])

    tm = MultiClassTsetlinMachine(100, 10.0, 2.0)

    start_training = time()
    tm.fit(x_train, y_train, epochs=10, incremental=True)
    stop_training = time()

    acc = 100 * (tm.predict(x_test) == y_test).mean()

    return acc, stop_training - start_training


def run_dnn(x_train, y_train, x_test, y_test):
    x_train = __preprocess_x(x_train).astype('float32')
    x_test = __preprocess_x(x_test).astype('float32')

    y_train = __preprocess_y(y_train)
    y_test = __preprocess_y(y_test)

    # compute the number of labels
    num_labels = len(np.unique(y_train))

    # convert to one-hot vector
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # network parameters
    batch_size = 128
    hidden_units = 17
    dropout = 0.45

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

    start_training = time()
    model.fit(x_train, y_train, epochs=20, batch_size=batch_size, verbose=0)
    stop_training = time()

    _, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    return acc, stop_training - start_training


def __preprocess_x(x):
    randx = np.random.choice([0, 1], size=x.shape)

    x = np.where(x == 'y', 1, x)
    x = np.where(x == 'n', 0, x)
    x = np.where(x == '?', randx, x)

    return x


def __preprocess_y(y):
    return np.where(y == 'republican', 1, 0)
