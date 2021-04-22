from typing import List, Dict, NamedTuple, Tuple, Union
from dateutil.parser import parse
from collections import Counter
import math, random
import matplotlib.pyplot as plt
from dataclasses import dataclass


from probability import inverse_norma_cdf
from statistics import correlation, standard_deviations
from linear_algebra import Matrix, Vector, make_matrix, distance, \
    vector_mean, subtract, magnitude, \
    scalar_multiply, dot
from gradient_descent import gradient_step

Vector = List[Union[int, float]]

def bucketize(point: float, bucket_size: float) -> float:
    """Floor the point to the next lower multiple of bucket_size

    Args:
        point (float): point ot be divided
        bucket_size (float): [description]

    Returns:
        float: [description]
    """
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    """Buckets the points and counts how many in each bucket

    Args:
        points (List[float]): data
        bucket_size (float): number of buckets

    Returns:
        Dict[Float, int]: Counter per bucket
    """

    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histgram(points: List[float], bucket_size: float, title: str = ""):
    """Plot histogram of data

    Args:
        points (List[float]): data
        bucket_size (float): number of buckets to dividedata into
        title (str, optional): Title of the plot. Defaults to "".
    """
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)

def random_normal() -> float:
    """Returns a random draw from a standard normal distribution

    Returns:
        float: [description]
    """

    return inverse_norma_cdf(random.random())

def correlation_matrix(data: List[Vector]) -> Matrix:
    """Returns the len(data) x len(data) matrix whose (i, j)-th entry
    is the correlation between data[i] and data[j]

    Args:
        data (List[Vector]): [description]

    Returns:
        Matrix: [description]
    """
    def correlation_ij(i: int, j: int) -> float:
        return correlation(data[i], data[j])
    return make_matrix(len(data), len(data), correlation_ij)

import datetime

# stock_price = {"closing_price": 102.06,
#                 "date": datetime.date(2014, 8, 29),
#                 "symbol": "AAPL"}

# stock_price["closing_price"] = 103.06

from collections import namedtuple

StockPrice = namedtuple("StockPrice", ["symbol", "date", "closing_price"])
price = StockPrice("MSFT", datetime.date(2018, 12, 14), 106.03)

class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        """It's a class, so we can add methods too"""
        return self.symbol in ["MSFT", "GOOG", "FB", "AMZN", "AAPL"]

price = StockPrice("MSFT", datetime.date(2018, 12, 14), 106.03)

@dataclass
class StockPrice2:
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> float:
        return self.symbol in ["MSFT", "GOOG", "FB", "AMZN", "AAPL"]

price2 = StockPrice2("MSFT", datetime.date(2018, 12, 14), 106.03)

def scale(data: List[List[float]]) -> Tuple[Vector, Vector]:
    """returns the mean and standard deviation for each position"""
    dim = len(data[0])
    means = vector_mean(data)
    stdevs = [standard_deviations([vector[i] for vector in data]) for i in range(dim)]
    return means, stdevs

def rescale(data: List[List[float]]) -> List[List[float]]:
    """Rescale the input data so that each position has
    mean 0 and standard deviation 1. (Leaves a position 
    as is if its standard deviation is 0.)

    Args:
        data (List[Vector]): input data

    Returns:
        List[Vector]: rescaled data
    """
    dim = len(data[0])
    rescaled = [v[:] for v in data]
    for v in rescaled:
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]
    return rescaled

def de_mean(data: List[Vector]) -> List[Vector]:
    """Recenters the data to have mean 0 in every dimension"""
    mean = vector_mean(data)
    return [subtract(vector, mean) for vector in data]

def direction(w: Vector) -> Vector:
    mag = magnitude(w)
    return [w_i / mag for w_i in w]
    magnitude()

def parse_row(row: List[str]) -> StockPrice:
    symbol, date, closing_price = row
    return StockPrice(symbol=symbol,
                        date=parse(date).date(),
                        closing_price=float(closing_price))

def directional_variance(data: List[Vector], w: Vector) -> float:
    """Returns the variance of x in the direction of w

    Args:
        data (List[Vector]): input data
        w (Vector): vector of direction

    Returns:
        float: directional variance of the input data
    """
    w_dir = direction(w)
    return sum(dot(v, w_dir) ** 2 for v in data)

def directional_variance_gradient(data: List[Vector], w: Vector) -> Vector:
    """The gradient of directional variacne with respect to w

    Args:
        data (List[Vector]): [description]
        w (Vector): [description]

    Returns:
        Vector: [description]
    """
    w_dir = direction(w)
    return [sum(2 * dot(v, w_dir) * v[i] for v in data) for i in range(len(w))]

def first_principal_component(data: List[Vector],
                                n: int = 100,
                                step_size: float = 0.1) -> Vector:
    """get first principal component

    Args:
        data (List[Vector]): [description]
        n (int, optional): [description]. Defaults to 100.
        step_size (float, optional): [description]. Defaults to 0.1.

    Returns:
        Vector: [description]
    """

    guess = [1.0 for _ in data[0]]

    with tqdm.trange(n) as t:
        for _ in t:
            dv = directional_variance(data, guess)
            gradient = directional_variance_gradient(data, guess)
            guess = gradient_step(guess, gradient, step_size)
            t.set_description(f"dv: {dv:.3f}")

    return direction(guess)

def project(v: Vector, w: Vector) -> Vector:
    """return the projection of v onto the direction w

    Args:
        v (Vector): [description]
        w (Vector): [description]

    Returns:
        Vector: [description]
    """
    project_length = dot(v, w)
    return scalar_multiply(project_length, w)

def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
    """projects v onto w and subtracts the results from v

    Args:
        v (Vector): vector to project
        w (Vector): vector of direction

    Returns:
        Vector: vector with projection removed
    """
    return subtract(v, project(v, w))

def remove_projection(data: List[Vector], w: Vector) -> List[Vector]:
    return [remove_projection_from_vector(v, w) for v in data]

def pca(data: List[Vector], num_components: int) -> List[Vector]:
    components: List[Vector] = []
    for _ in range(num_components):
        component = first_principal_component(data)
        components.append(component)
        data = remove_projection(data, component)
    return components
    
def transform_vector(v: Vector, components: List[Vector]) -> Vector:
    return [dot(v, w) for w in components]

def transform(data: List[Vector], components: List[Vector]) -> List[Vector]:
    return [transform_vector(v, components) for v in data]



# Now test our function
stock = parse_row(["MSFT", "2018-12-14", "106.03"])

assert stock.symbol == "MSFT"
assert stock.date == datetime.date(2018, 12, 14)
assert stock.closing_price == 106.03


if __name__ == "__main__":
    random.seed(0)

    # uniform between -100 and 100
    uniform = [200 * random.random() - 100 for _ in range(10000)]

    # normal distribution with mean 0, standard deviation 57
    normal = [57 * inverse_norma_cdf(random.random()) for _ in range(10000)]

    plot_histgram(uniform, 10, "Uniform Histogram")

    plot_histgram(normal, 10, "Normal Histogram")

    # Test random normal
    xs = [random_normal() for _ in range(1000)]
    ys1 = [x + random_normal() / 2 for x in xs]
    ys2 = [-x + random_normal() / 2 for x in xs]

    # plt.scatter(xs, ys1, marker=".", color="black", label="ys1")
    # plt.scatter(xs, ys2, marker=".", color="gray", label="ys2")
    # plt.xlabel("xs")
    # plt.ylabel("ys")
    # plt.legend(loc=9)
    # plt.title("Very different joint distributions")
    # plt.show()

    print(correlation(xs, ys1))
    print(correlation(xs, ys2))

    assert price.symbol == "MSFT"
    assert price.closing_price == 106.03
    assert price.is_high_tech()

    assert price2.symbol == "MSFT"
    assert price2.closing_price == 106.03
    assert price2.is_high_tech()

    vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
    means, stdevs = scale(vectors)
    assert means == [-1, 0, 1]
    assert stdevs == [2, 1, 0]

    means, stdevs = scale(rescale(vectors))
    assert means == [0, 0, 1]
    assert stdevs == [1, 1, 0]
    

    import tqdm

    # for i in tqdm.tqdm(range(100)):
    #     _ = [random.random() for _ in range(1000000)]

    def primes_up_to(n: int) -> List[int]:
        primes = [2]

        with tqdm.trange(3, n) as t:
            for i in t:
                i_is_prime = not any(i % p == 0 for p in primes)
                if i_is_prime:
                    primes.append(i)
                
                t.set_description(f"{len(primes)} primes")

        return primes
    
    my_primes = primes_up_to(10000)

