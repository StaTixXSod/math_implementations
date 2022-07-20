import math
import numpy as np
import matplotlib.pyplot as plt

def mean(vector: list) -> float:
    """Return the mean value of the vector.

    -----
    INFO:
    -----
    The mean value is the sum of all values, divided by its length.
    Also, the mean value called as the expected value (E(x)).

    --------
    FORMULA:
    --------
    E(x) = (x1 + x2 + ... + xn) / NO

        where:
            x1...xn: vector values
            NO: number of observations
    -----
    Args:
        vector (list): value vector

    Returns:
        float: the mean value
    """
    m = 0
    no = len(vector)
    for item in vector:
        m += item
    return m / no

def var(vector: list) -> float:
    """Return variance of the vector

    -----
    INFO:
    -----
    The variance shows, how much values in vector deviates from its mean value in average.
    The deviation can be positive or negative. Because of that, to transform negative values to positive use squaring.

    Lower the variance, the closer the values are to each other.

    --------
    FORMULA:
    --------
    VAR(x) = E((x - E(x))^2)
        
        where
            E(x): mean value of x vector
            x - E(x): difference between x and mean value
    -----
    Args:
        vector (list): value vector

    Returns:
        float: variance
    """
    E = mean(vector)
    var = []
    for x in vector:
        var.append((x - E)**2)
    return mean(var)

def std(vector: list) -> float:
    """Return Standard Deviation of vector

    ------
    INFO:
    ------
    The standard deviation is the same as a variance,
    but here, to get STD, we calculate the square root
    of the variance, to get rid of squaring in VAR function.

    Denoted as the sigma symbol.

    --------
    FORMULA:
    --------  
    STD(x) = SQRT(VAR(x))

    -----
    Args:
        vector (list): value vector

    Returns:
        float: standart deviation value
    """
    return var(vector)**0.5

def standard_error(vector: list) -> float:
    """Return the standard error of the vector

    FORMULA:
    --------
    SE = STD / SD
        where:
        STD: Standard deviation,
        SD: Sampling distribution ( SQRT(NO) )
        NO: Number of observations

    Args:
        vector (list): Array

    Returns:
        float: Standard error value
    """
    sdev = std(vector)
    sdist = len(vector)**0.5
    return sdev / sdist

def t_value(popmean: float, samplemean: float, sd: float, no: int) -> float:
    se = sd / no**0.5
    z = (popmean - samplemean) / se
    return z

def paired_ttest_simp(m1: float, m2: float, sd1: float, sd2: float, n1: int, n2: int) -> float:
    se1 = sd1**2 / n1
    se2 = sd2**2 / n2

    t = (m1 - m2) / (se1 + se2)**0.5

    return t

def percentile(v: list, percent: float) -> float:
    """Return the percentile value of an array

    NOTE:
    -----
    - Percentile of an array = 0.5 is the median.
    - Percentile of an array = 1.0 is the CDF (Cummulative Distribution Function).

    Args:
        v (list): Array
        percent (float): percentage value from 0 to 1

    Returns:
        float: Percentage value
    """
    v = sorted(v)
    k = (len(v) - 1) * percent

    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return v[f]
    else:
        l = v[f]
        h = v[c]
        return (l + h) / 2

def median(vector: list) -> float:
    """Return the median value of list"""
    return percentile(vector, 0.5)

def qqplot(vector: list):
    """Plot Q-Q graph

    INFO:
    -----
    The Q-Q plot shows the distribution of a given vector,
    relative to the standard normal distriburion. The main reason to use Q-Q plot is to find out
    if the sample data has a normal distribution, because if we know the data is normally distributed, 
    we can assume some theories and run some tests.

    The Q-Q plot is better to use, when we have not too much data.

    INTERPRETATION:
    ---------------
    1. While the points of sample quantiles lies on the line, this means, that the
    sample points fits to the standard normal distribution.
    2. If the points are above the line, this means we are getting too high results,
    than we have to get.
    3. Otherwise, if the points are below the line, this means we are getting too low results,
    than we have to get, if we our sample data have normal distribution.
     

    Args:
        vector (list): 1D vector.

    Returns:
        plt.plot: Q-Q plot
    """
    norm_dist = np.random.standard_normal(len(vector))

    x = np.percentile(norm_dist, range(100))
    y = np.percentile(vector, range(100))

    plt.figure(figsize=(8, 8))
    plt.style.use("ggplot")
    plt.scatter(x, y, lw=2, label="Diff of a given vector from normal distribution", c='b')
    plt.plot(x, x, label="Normal distribution", c='r')
    plt.xlabel("Theoretical Quatiles")
    plt.ylabel("Sample Quatiles")
    plt.legend()
    plt.title("Q-Q plot")
    plt.show()

def flatten(v: list):
    """Return flatten list"""
    lst = []
    for i in range(len(v)):
        for item in v[i]:
            lst.append(item)
    return lst