import random

from linear_algebra import Vector, dot, distance, add, scalar_multiply
from typing import Callable, List, Tuple

def sum_of_squares(v: Vector) -> float:
    """Computes the sum of squared elements in v"""
    return dot(v, v)

def difference_quotient(f: Callable[[float], float],
                            x: float,
                            h: float) -> float:
    return (f(x + h) - f(x)) / h

def square(x: float) -> float:
    return x*x

def derivative(x: float) -> float:
    return 2*x

def partial_difference_quotient(f: Callable[[Vector], float],
                                    v: Vector,
                                    i: int,
                                    h: float) -> float:
    """Returns the i-th partial difference quotient of f at v

    Args:
        f (Callable[[Vector], float]): [description]
        v (Vector): input vector
        i (int): the index of the variable to differentiate
        h (float): distance

    Returns:
        float: partial quotient 
    """
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[Vector], float],
                        v: Vector,
                        h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves 'step_size' in the 'gradient' direction from 'v'

    Args:
        v (Vector): [description]
        gradient (Vector): [description]
        step_size (float): [description]

    Returns:
        Vector: [description]
    """


if __name__ == "__main__":

    xs = range(-10, 11)
    actuals = [derivative(x) for x in xs]
    estimates = [difference_quotient(square, x, h=0.0001) for x in xs]
    import matplotlib.pyplot as plt

    plt.title("Actual Derivatives vs. Estimates")
    plt.plot(xs, actuals, "rx", label="Actual")
    plt.plot(xs, estimates, "b+", label="Estimates")