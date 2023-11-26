from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np

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


def preprocess_tsetlin(x, y):
    return x, __preprocess_y(y)


def train_tsetlin(x, y, num_clauses=250, T=10, s=1, epochs=50):
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


def train_dnn(x, y, batch_size=50, hidden_units=120, dropout=0.3, epochs=120):
    categorical_encoder = OneHotEncoder(categories=CATEGORIES)
    categorical_encoder.fit(x[:, CATEGORICAL_COLUMNS])

    encode_x = lambda data: np.concatenate(
        (categorical_encoder.transform(data[:, CATEGORICAL_COLUMNS]).toarray(),
         data[:, NUMERIC_COLUMNS]), axis=1).astype('float32')
    x = encode_x(x)

    # compute the number of labels
    num_labels = len(np.unique(y))

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
    return np.where(y == '>50K', 1, 0)


def replace(col, catz):
    for i in range(0, len(col)):
        col[i] = catz.index(col[i])
    return col


def categorical_to_int(X):
    for i in range(0, len(CATEGORIES)):
        col_num = CATEGORICAL_COLUMNS[i]
        X[:, col_num] = replace(X[:, col_num], CATEGORIES[i])
    return X
