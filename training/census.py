from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np

from time import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder

CATEGORIES = [
    ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay",
     "Never-worked", "?"],
    ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th",
     "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
    ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent",
     "Married-AF-spouse"],
    ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
     "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
     "Priv-house-serv", "Protective-serv", "Armed-Forces", "?"],
    ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    ["Female", "Male"],
    ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)",
     "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland",
     "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador",
     "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
     "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "?"]
]

CATEGORICAL_COLUMNS = [1, 3, 5, 6, 7, 8, 9, 13]
NUMERIC_COLUMNS = [2, 4, 10, 11, 12]


def run_tsetlin(x_train, y_train, x_test, y_test):
    x_train, x_test = __preprocess_x(x_train), __preprocess_x(x_test)
    y_train, y_test = __preprocess_y(y_train), __preprocess_y(y_test)

    tm = MultiClassTsetlinMachine(250, 10, 1)

    start_training = time()
    tm.fit(x_train, y_train, epochs=50, incremental=True)
    stop_training = time()

    acc = 100 * (tm.predict(x_test) == y_test).mean()

    return acc, stop_training - start_training


def run_dnn(x_train, y_train, x_test, y_test):
    x_train, x_test = __preprocess_x(x_train), __preprocess_x(x_test)
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

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
    return np.where(y == '>50K', 1, 0)


def __preprocess_x(x):
    categoricals = x[:, CATEGORICAL_COLUMNS]
    categoricals_one_hot = OneHotEncoder(categories=CATEGORIES).fit(categoricals).transform(categoricals)

    numerics = x[:, NUMERIC_COLUMNS]
    discretizer = KBinsDiscretizer(encode="onehot",
                                   strategy="quantile",
                                   n_bins=5)
    discretizer.fit(numerics)

    numerics_discretized = discretizer.transform(numerics)

    return np.concatenate((categoricals_one_hot.toarray(), numerics_discretized.toarray()), axis=1)
