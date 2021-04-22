import logging
import random
from typing import TypeVar, List, Tuple

from logzero import setup_logger
from config import CONFIG

loglvl = dict(info=logging.INFO, debug=logging.DEBUG, warning=logging.WARNING)


X = TypeVar("X")

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]

    Args:
        data (List[X]): input data
        prob (float): split ratio

    Returns:
        Tuple[List[X], List[X]]: slitted data
    """
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]

Y = TypeVar("Y")

def train_test_split(xs: List[X],
                    ys: List[Y],
                    test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    """Generate indices and split input data

    Args:
        xs (List[X]): input data
        ys (List[Y]): target variable
        test_pct (float): split ratio

    Returns:
        Tuple[List[X], List[X], List[Y], List[Y]]: [description]
    """
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)
    return ([xs[i] for i in train_idxs],
             [xs[i] for i in test_idxs],
             [ys[i] for i in train_idxs],
             [ys[i] for i in test_idxs])


def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    """calculate accuracy score

    Args:
        tp (int): true positives
        fp (int): false positives
        fn (int): false negatives
        tn (int): true negatives

    Returns:
        float: acc
    """
    correct = tp + tn
    total = tp + tn + fn + fp
    return correct / total

def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    """calculate precision score

    Args:
        tp (int): true positives
        fp (int): false positives
        fn (int): false negatives
        tn (int): true negatives

    Returns:
        float: precision
    """
    return tp / (tp + fp)

def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    """compute recall 

    Args:
        tp (int): true positives
        fp (int): false positives
        fn (int): false negatives
        tn (int): true negatives

    Returns:
        float: [description]
    """
    return tp / (tp + fn)

def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    """compute f1-score

    Args:
        tp (int): true positives
        fp (int): false positives
        fn (int): false negatives
        tn (int): true negatives

    Returns:
        float: f1 score
    """
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)


if __name__ == "__main__":

    logger = setup_logger(__file__, level=logging.DEBUG, logfile=str(CONFIG.report / "ml.log"))
    pass

