from ml import *

def test_split_data():
    data = [n for n in range(1000)]
    train, test = split_data(data, .75)
    assert len(train) == 750
    assert len(test) == 250
    assert sorted(train + test) == data

def test_train_test_split():
    xs = [x for x in range(1000)]
    ys = [2 * x for x in xs]
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

    assert len(x_train) == len(y_train) == 750
    assert len(x_test) == len(y_test) == 250

    assert all(y == 2 * x for x, y in zip(x_train, y_train))
    assert all(y == 2 * x for x, y in zip(x_test, y_test))

def test_accuracy():
    assert accuracy(70, 4930, 13930, 981070) == 0.98114

def test_precision():
    assert precision(70, 4930, 13930, 981070) == 0.014

def test_recall():
    assert recall(70, 4930, 13930, 981070) == 0.005
