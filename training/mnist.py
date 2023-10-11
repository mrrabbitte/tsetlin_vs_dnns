
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical


def run_tsetlin(x_train, y_train, x_test, y_test):
    x_train = np.where(x_train.reshape((x_train.shape[0], 28 * 28)) > 75, 1, 0)
    x_test = np.where(x_test.reshape((x_test.shape[0], 28 * 28)) > 75, 1, 0)

    tm = MultiClassTsetlinMachine(100, 10.0, 2.0)

    start_training = time()
    tm.fit(x_train, y_train, epochs=10, incremental=True)
    stop_training = time()

    acc = 100 * (tm.predict(x_test) == y_test).mean()

    return acc, stop_training - start_training


def run_dnn(x_train, y_train, x_test, y_test):
    # Binarize the data as well
    # x_train = np.where(x_train > 75, 1, 0)
    # x_test = np.where(x_test > 75, 1, 0)

    # compute the number of labels
    num_labels = len(np.unique(y_train))

    # convert to one-hot vector
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    image_size = x_train.shape[1]
    input_size = image_size * image_size

    x_train = np.reshape(x_train, [-1, input_size])
    x_train = x_train.astype('float32') / 255
    x_test = np.reshape(x_test, [-1, input_size])
    x_test = x_test.astype('float32') / 255

    # network parameters
    batch_size = 128
    hidden_units = 256
    dropout = 0.45

    model = Sequential()
    model.add(Dense(hidden_units, input_dim=input_size))
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
