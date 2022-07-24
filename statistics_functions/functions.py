import math
import numpy as np
import matplotlib.pyplot as plt

def mean(vector: list, ddof: int = None) -> float:
    """Return the mean value of the vector.

    Info:
    -----
    The mean value is the sum of all values, divided by its length.
    Also, the mean value called as the expected value (E(x)).

    Formula:
    --------
    `E(x) = (x1 + x2 + ... + xn) / NO`

        where:
            x1...xn: vector values
            NO: number of observations
    -----
    Args:
        vector (list): value vector
        ddof (int): degrees of freedom

    Returns:
        float: the mean value
    """
    m = 0
    if ddof == None:
        no = len(vector)
    else:
        no = ddof
    for item in vector:
        m += item
    return m / no

def var(vector: list, ddof: int = None) -> float:
    """Return variance of the vector

    Info:
    -----
    The variance shows, how much values in vector deviates from its mean value in average.
    The deviation can be positive or negative. Because of that, to transform negative values to positive use squaring.

    Lower the variance, the closer the values are to each other.

    Formula:
    --------
    `VAR(x) = E((x - E(x))^2)`
        
        where
            E(x): mean value of x vector
            x - E(x): difference between x and mean value
    -----
    Args:
        vector (list): value vector
        ddof (int): degrees of freedom

    Returns:
        float: variance
    """
    E = mean(vector, ddof=ddof)

    var = []
    for x in vector:
        var.append((x - E)**2)
    return mean(var, ddof)

def std(vector: list, ddof: int = None) -> float:
    """Return Standard Deviation of vector

    Info:
    -----
    The standard deviation is the same as a variance,
    but here, to get STD, we calculate the square root
    of the variance, to get rid of squaring in VAR function.

    Denoted as the sigma symbol.

    Formula:
    --------  
    `STD(x) = SQRT(VAR(x))`

    -----
    Args:
        vector (list): value vector
        ddof (int): degrees of freedom

    Returns:
        float: standart deviation value
    """
    return var(vector, ddof=ddof)**0.5

def standard_error(vector: list) -> float:
    """Return the standard error of the vector

    Formula:
    --------
    `SE = STD / SD`
        where:
        STD: Standard deviation,
        SD: Sampling distribution ( `SQRT(NO)` )
        NO: Number of observations
    
    -----
    Args:
        vector (list): Array

    Returns:
        float: Standard error value
    """
    sdev = std(vector, len(vector)-1)
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

    Note:
    -----
    - Percentile of an array = 0.5 is the median.
    - Percentile of an array = 1.0 is the CDF (Cummulative Distribution Function).
    
    -----
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

    Interpretation:
    ---------------
    1. While the points of sample quantiles lies on the line, this means, that the
    sample points fits to the standard normal distribution.
    2. If the points are above the line, this means we are getting too high results,
    than we have to get.
    3. Otherwise, if the points are below the line, this means we are getting too low results,
    than we have to get, if we our sample data have normal distribution.
     
    -----
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

def flatten(v: list) -> list:
    """Return flatten list"""
    return [i for j in v for i in j]

def paired_diff(a: list, b: list) -> list:
    """Return a list with a pairwise difference of elements"""
    assert len(a) == len(b)
    return [ai - bi for ai, bi in zip(a, b)]

def covariation(x: list, y: list) -> float:
    """Return covariation value

    Info:
    -----
    It is like the VARiance, but for 2 arrays, that shows the value of relationship between two samples of data.
    If the data has positive relationship, the covariation value will be positive, otherwise negative
    (or approximately will be 0). To calculate the relationship between 2 samples, we calculate 
    mean value for X data and Y data. If most samples of data located higher X mean and Y mean AND 
    lower X mean and Y mean, this mean our data has POSITIVE correlation (covariation), because 
    points are located in left lower square and upper right square.

    Example:
    --------
    * +------+------+
    * |oooooo|xxxxxx|
    * |oooooo|xxxxxx|
    * +------+------+
    * |xxxxxx|oooooo|
    * |xxxxxx|oooooo|
    * +------+------+

    - If we imagine, that we have only "x" values, then relationship will be POSITIVE
    - If only "o" values, then NEGATIVE relationship

    Steps:
    ------
    1. Calculate the mean values for x and y datas (X and Y)
    2. For each element in x and y calculate the difference between (xi, X) and (yi, Y)
    3. Multiply the X difference and Y difference, if X diff is positive (+), 
    but Y diff is negative (-), then the general difference will be negative. 
    (++ = +; -+ = -; +- = -; -- = +)
    4. Calculate the mean difference, that will be 'covariance' value

    Formula:
    --------
    `cov = MEAN( (xi - X) * (yi - Y) )`
    
    where:
        xi, yi: each value in data
        X, Y: mean values for x and y data

    Args:
        x, y (list): array lists of data to compare

    Returns:
        float: covariation value
    """
    assert len(x) == len(y)
    X, Y = mean(x), mean(y)
    cov = [(xi - X) * (yi - Y) for xi, yi in zip(x, y)]
    cov_mean = mean(cov, ddof=len(x)-1)
    return cov_mean

def correlation(x: list, y: list, formula: str="1") -> float:
    """Return correlation coefficient

    Info:
    -----
    Correlation coefficient, also called Pearson Correlation coefficient 
    shows the relationship between 2 samples of data. This is the covariation value,
    divided by product of std(x) and std(y) to transform the value 
    into range from -1 to 1.

    Formula:
    --------
    * `r_xy = cov / (std(x) * std(y))`

        where
            r_xy: correlation coefficient
            cov: covariation value
    
    * `cov(x, y) = E[(xi - X) * (yi - Y)]`

    Formula 1:
    ----------
    The formula for 'r' can also be written as:

    `r_xy = num / den`
    
    where:
    * `num = E[xy] - E[x] * E[y]`
    * `den = SQRT((E[x^2] - E[x])^2 * (E[y^2] - E[y])^2)`

    Formula 2:
    ----------
    * `cov(x, y) = SUM((xi - X) * (yi - Y))`
    * `sdev = SQRT( SUM((xi - X)^2) * SUM((yi - Y)^2) )`
    * `r_xy = cov / sdev`

    -----
    Args:
        x, y (list): array list of data to compare

    Returns:
        float: correlation coefficient value
    """
    assert len(x) == len(y)
    xy = mean(paired_prod(x, y))
    X, Y = mean(x), mean(y)
    
    if formula == "1":
        num = xy - X * Y
        x_var = mean(squared(x)) - mean(x)**2
        y_var = mean(squared(y)) - mean(y)**2
        den = (x_var * y_var)**0.5
        return num / den
    
    elif formula == "2":
        cov = sum([(xi - X) * (yi - Y) for xi, yi in zip(x, y)])
        xvar = sum([(xi - X)**2 for xi in x])
        yvar = sum([(yi - Y)**2 for yi in y])
        sdev = (xvar * yvar)**0.5
        return cov / sdev

def paired_prod(x: list, y: list) -> list:
    """Return a list of pairwise multiplications"""
    return [xi*yi for xi, yi in zip(x, y)]
    
def squared(a: list) -> list:
    """Return the same list with squared values"""
    return [ai**2 for ai in a]