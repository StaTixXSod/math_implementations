import os
import sys

sys.path.append(os.getcwd())
from statistics_functions.functions import *

def one_sample_ttest(v1, v2) -> float:
    """Return T test value for population data
    and sample data

    INFO:
    -----
    This test compare the population and the sample data. 
    Usually use, when we have general data and we want to know, 
    if our sample mean belongs to general mean.

    This test shows, how far our sample mean (X) deviates from 
    the population data mean (M).

    FORMULA:
    --------
    >>> t = (M - X) / (SE)
        where:
            M: population mean
            X: sample mean
            SE: Standard Error of sample data

    Here we use SE of sample data, because usually we can't 
    calculate SE for population data, because we don't know the STD of population data.

    Args:
        population (list): population data
        sample (list): sample data
    """
    pop_mean = mean(v1)
    sample_mean = mean(v2)

    t = (pop_mean - sample_mean) / standard_error(v2)
    return t

def two_sample_ttest(v1, v2):
    pass

def paired_ttest(v1: list, v2: list) -> float:
    """Return T Student Criterion statistic value
    
    INFO:
    -----
    This criterion allows to compare 2 samples of data
    and called "Paired TTest" or "T-Student Criterion"

    Hypothesis:
    The null hypothesis (H0): there is no difference between this 2 sample means
        H0: M1 = M2
    The alternative hypothesis (H1): this 2 sample means NOT EQUAL to each other
    and there is a good chances to get some great results
        H1: M1 != M2

    It would be good to get H1.

    FORMULA:
    --------
    t = (X1 - X2) - (M1 - M2) / SQRT( (sd1^2 / n1) + (sd2^2 / n2) )

    Suppose that M1 - M2 = 0, because we assume in population data M1 and M2 are identical,
    so we can rewrite the formula:

    t = (X1 - X2) / SQRT( (sd1^2 / n1) + (sd2^2 / n2) )

    INTERPRETATION:
    ---------------
    This value says, that our difference between 2 sample means deviates 
    from the population mean on "t" sigma value.

    NOTE:
    -----
    1. To use T Test, it is better that variance from this 2 samples
    were approximately the same (dispersion homogeneity requirement). 
    To check this requirement use "Levene test" or "Fisher criterion".
    2. If NO (number of observations) of sample data is not really big (NO < 30), this is very important
    to have normal distribution of sample data. If NO > 30, this criterion is fine
    even without normal distribution. (But it's steel better to have normal distribution)

    Args:
        v1 (list): 1st list of observations
        v2 (list): 2nd list of observations
    """
    X1, X2 = mean(v1), mean(v2)
    sd1, sd2 = std(v1), std(v2)
    n1, n2 = len(v1), len(v2)

    se1 = sd1**2 / n1
    se2 = sd2**2 / n2

    t = (X1 - X2) / (se1 + se2)**0.5

    return t
