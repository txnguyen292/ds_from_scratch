from typing import List
from collections import Counter
import math
from linear_algebra import sum_of_squares, dot

Vector = List[float]

def mean(xs: Vector) -> float:
    return sum(xs) / len(xs)

def _median_odd(xs: Vector) -> float:
    """if len(xs) is odd, the median is the middle element"""
    return sorted(xs)[len(xs) // 2]

def _median_even(xs: Vector) -> float:
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2

def median(v: Vector) -> float:
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

def quantile(xs: Vector, p: float) -> float:
    """Returns the pth-percentile value in x"""
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]

def mode(x: Vector) -> Vector:
    """Returns a list, since there might be more than one mode"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]

def data_range(xs: Vector) -> float:
    return max(xs) - min(xs)

def de_mean(xs: Vector) -> Vector:
    """Translate xs by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: Vector) -> float:
    """Almost the average squared deviation from the mean"""
    assert len(xs) >= 2, "Variance requires at least two elements"

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)

def standard_deviations(xs: Vector) -> float:
    """The standard deviation is the square root of the variance"""
    return math.sqrt(variance(xs))

def interquartile_range(xs: Vector) -> float:
    """Returns the difference between the 75th and 25th percentile"""
    return quantile(xs, 0.75) - quantile(xs, 0.25)

def covariance(xs: Vector, ys: Vector) -> float:
    """Returns the covariance of two vectors"""

    assert len(xs) == len(ys), "xs and ys must be of same length"
    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)


def correlation(xs: Vector, ys: Vector) -> float:
    """Measures how much xs and ys vary in tandem about their means"""
    stdev_x = standard_deviations(xs)
    stdev_y = standard_deviations(ys)

    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0



if __name__ == "__main__":
    # Test median
    assert median([1, 10, 2, 9, 5]) == 5
    assert median([1, 9, 2, 10]) == (2 + 9) / 2
    # Test quantile
    # Later