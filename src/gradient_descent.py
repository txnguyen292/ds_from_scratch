import random

from linear_algebra import Vector, dot, distance, add, scalar_multiply, vector_mean
from typing import Callable, List, Tuple, TypeVar, Iterable

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
                        h: float = 0.0001) -> Vector:
    """estimate the gradient

    Args:
        f (Callable[[Vector], float]): [description]
        v (Vector): [description]
        h (float, optional): [description]. Defaults to 0.0001.

    Returns:
        [type]: Vector of partial gradients
    """
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves 'step_size' in the 'gradient' direction from 'v'

    Args:
        v (Vector): input vector
        gradient (Vector): gradient of the input vector
        step_size (float): number of steps to move

    Returns:
        Vector: vector after moving 'step_size' in the direction of the gradient
    """

    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    """sum of squares of a vector

    Args:
        v (Vector): input vector

    Returns:
        Vector: sum of squared vector
    """
    return [2*v_i for v_i in v]

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    """Computes gradients for linear function

    Args:
        x (float): [description]
        y (float): [description]
        theta (Vector): [description]

    Returns:
        Vector: [description]
    """
    slope, intercept = theta
    predicted = slope * x + intercept
    error = (predicted - y)
    squared_error = error * 2
    grad = [2 * error * x, 2 * error]
    return grad

# Minibatch and Stochastic gradient descent

T = TypeVar("T") # generic functions

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool=True) -> Iterable[List[T]]:
    """Generate "batch_size"-sized minibatches from the dataset

    Args:
        dataset (List[T]): input data
        batch_size (int): size of each minibatch
        shuffle (bool, optional): whether to shuffle the data. Defaults to True.

    Yields:
        Iterator[List[T]]: minibatch of dat
    """

    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts) # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]




if __name__ == "__main__":

    xs = range(-10, 11)
    actuals = [derivative(x) for x in xs]
    estimates = [difference_quotient(square, x, h=0.0001) for x in xs]
    import matplotlib.pyplot as plt

    plt.title("Actual Derivatives vs. Estimates")
    plt.plot(xs, actuals, "rx", label="Actual")
    plt.plot(xs, estimates, "b+", label="Estimates")

    # Test sum_of_squares gradient
    v = [random.uniform(-10, 10) for i in range(3)]
    for epoch in range(1000):
        grad = sum_of_squares_gradient(v)
        v = gradient_step(v, grad, -0.01)
        print(epoch, v)
    assert distance(v, [0, 0, 0]) < 0.001

    # batch gradient for model
    inputs = [(x, 20 * x + 5) for x in range(-50, 50)] # generating input data

    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    learning_rate = 0.001

    for epoch in range(5000):
        # Compute the mean of the gradients
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
        # Take a step in that directon
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

    slope, intercept = theta

    assert 19.9 < slope < 20.1, "slope should be about 20"
    assert 4.9 < intercept < 5.1, "intercept should be about 5"

    # minibatch gradient descent
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

    for epoch in range(1000):
        for batch in minibatches(inputs, batch_size=20):
            grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

    slope, intercept = theta
    assert 19.9 < slope < 20.1, "slope should be about 20"
    assert 4.9 < intercept < 5.1, "intercept should be about 5"

    # stochastic gradient descent
    for epoch in range(100):
        for x, y in inputs:
            grad = linear_gradient(x, y, theta)
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

    slope, intercept = theta
    assert 19.9 < slope < 20.1, "slope should be about 20"
    assert 4.9 < intercept < 5.1, "intercept should be about 5"