import enum, random, math

class Kid(enum.Enum):
    BOY = 0
    GIRL = 1

def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])

both_girls = 0
older_girl = 0
either_girl = 0

def uniform_pdf(x: float) -> float:
    return 1 if 0 <= x < 1 else 0

def uniform_cdf(x: float) -> float:
    """Returns the probability that a uniform random variable is <= x

    Args:
        x (float): [description]

    Returns:
        float: [description]
    """

    if x < 0: return 0
    elif x < 1: return x
    else: return 1

SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float=0, sigma: float=1) -> float:
    """implement the normal pdf

    Args:
        x (float): x random variable
        mu (float, optional): mean. Defaults to 0.
        sigma (float, optional): variance. Defaults to 1.

    Returns:
        float: normal random variable
    """
    return (math.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))

def normal_cdf(x: float, mu: float = 0, sigma: float=1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def inverse_norma_cdf(p: float,
                        mu: float=0,
                        sigma: float=1,
                        tolerance: float=0.0001) -> float:
    """Find approximate inverse using binary search

    Args:
        p (float): [description]
        mu (float, optional): [description]. Defaults to 0.
        sigma (float, optional): [description]. Defaults to 1.
        tolerance (float, optional): [description]. Defaults to 0.0001.

    Returns:
        float: [description]
    """
    # if not stander, compute standerf and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_norma_cdf(p, tolerance=tolerance)

    low_z = -10.0
    hi_z = 10.0
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z
    return mid_z

def bernoulli_trial(p: float) -> int:
    """Returns 1 with probability p and 0 with probability 1-p

    Args:
        p (float): [description]

    Returns:
        int: [description]
    """
    return 1 if random.random() < p else 0


def binomial(n: int, p: float) -> int:
    """Returns the sum of n bernoully(p) trials

    Args:
        n (int): [description]
        p (float): [description]

    Returns:
        int: [description]
    """
    return sum(bernoulli_trial(p) for _ in range(n))
if __name__ == "__main__":

    random.seed(0) 
    for _ in range(10000):
        younger = random_kid()
        older = random_kid()

        if older == Kid.GIRL:
            older_girl += 1
        if older == Kid.GIRL and younger == Kid.GIRL:
            both_girls += 1
        if older == Kid.GIRL or younger == Kid.GIRL:
            either_girl += 1

    print(f"P(both | older): {both_girls / older_girl}")
    print(f"P(both | either): {both_girls / either_girl}")



    # Test normal pdf
    import matplotlib.pyplot as plt
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
    plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
    plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
    plt.plot(xs,[normal_pdf(x,mu=-1)   for x in xs],'-.',label='mu=-1,sigma=1')
    plt.legend()
    plt.title("Various Normal pdfs")
    plt.show()

    # test normal cdfs
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
    plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
    plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
    plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
    plt.legend(loc=4) # bottom right
    plt.title("Various Normal cdfs")
    plt.show()

    # Test bernoulli
    def binomial_historgram(p: float, n: int, num_points: int) -> None:
        """Picks points from a Binomial(n, p) and plots their histograms

        Args:
            p (float): [description]
            n (int): [description]
            num_points (int): [description]
        """

        data = [binomial(n, p) for _ in range(num_points)]
        histogram = Counter(data)
        plt.bar([x - 0.4 for x in historgram.keys()],
                [v / num_points for v in histogram.values()],
                    0.8, color="0.75")
        mu = p * n
        sigma = math.sqrt(n * p * (1 - p))

        xs = range(min(data), max(data) + 1)
        ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]

        plt.plot(xs, ys)
        plt.title("Binomial Distribution vs. Normal Approximation")
        plt.show()