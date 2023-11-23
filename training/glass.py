from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical

from sklearn.preprocessing import KBinsDiscretizer


def preprocess_tsetlin(x, y):
    return x, y


def train_tsetlin(x_train, y_train, num_clauses=200, T=10, s=5.0, epochs=10):
    discretizer = KBinsDiscretizer(encode="onehot-dense", strategy="quantile", n_bins=4)
    discretizer.fit(x_train)

    x_train = discretizer.transform(x_train)

    tm = MultiClassTsetlinMachine(num_clauses, T, s)
    tm.fit(x_train, y_train, epochs=epochs, incremental=True)

    return lambda x_test: tm.predict(discretizer.transform(x_test))


def preprocess_dnn(x, y):
    return x.astype('float32'), to_categorical(y)


def train_dnn(x, y, batch_size=50, hidden_units=120, dropout=0.00001, epochs=120):
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

    return lambda x_test: model.predict(x_test, batch_size=batch_size)

