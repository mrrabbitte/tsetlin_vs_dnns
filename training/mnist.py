
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical


def preprocess_tsetlin(x, y):
    return np.where(x.reshape((x.shape[0], 28 * 28)) > 75, 1, 0), y


def train_tsetlin(x_train, y_train, num_clauses=100, T=10, s=2.0, epochs=10):
    tm = MultiClassTsetlinMachine(num_clauses, T, s)

    tm.fit(x_train, y_train, epochs=epochs, incremental=True)

    return lambda x_test: tm.predict(x_test)


def preprocess_dnn(x, y):
    image_size = x.shape[1]
    input_size = image_size * image_size

    x = np.reshape(x, [-1, input_size])
    x = x.astype('float32') / 255

    return x,  to_categorical(y)


def train_dnn(x_train, y_train, batch_size=128, hidden_units=256, dropout=0.45, epochs=20):
    num_labels = 10

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

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    return lambda x_test: model.predict(x_test, batch_size=batch_size)

