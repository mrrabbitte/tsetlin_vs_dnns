
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np

from time import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical


def run_tsetlin(x_train, y_train, x_test, y_test):
    y_train, y_test = __preprocess_y(y_train), __preprocess_y(y_test)

    tm = MultiClassTsetlinMachine(250, 10, 1)

    start_training = time()
    tm.fit(x_train, y_train, epochs=50, incremental=True)
    stop_training = time()

    acc = 100 * (tm.predict(x_test) == y_test).mean()

    return acc, stop_training - start_training


def run_dnn(x_train, y_train, x_test, y_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train, y_test = __preprocess_y(y_train), __preprocess_y(y_test)

    # compute the number of labels
    num_labels = len(np.unique(y_train))

    # convert to one-hot vector
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # network parameters
    batch_size = 50
    hidden_units = 120
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

    start_training = time()
    model.fit(x_train, y_train, epochs=120, batch_size=batch_size, verbose=0)
    stop_training = time()

    _, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    return acc, stop_training - start_training


def __preprocess_y(y):
    return np.where(y == 'M', 1, 0)
