import os
import sys

sys.path.append(os.getcwd())
from typing import Tuple
from statistics_functions.functions import *
from scipy.stats import t


def one_sample_ttest(a: list, popmean: float) -> Tuple[float, float]:
    """Return T test statistic value for population data
    and sample data

    Info:
    -----
    This test compare the population and the sample data. 
    Usually use, when we have general data and we want to know
    if our sample mean belongs to general mean.

    This test shows, how far our sample mean (X) deviates from 
    the population data mean (M).

    Formula:
    --------
    `t = (M - X) / (SE)`
        where:
            M: population mean
            X: sample mean
            SE: Standard Error of sample data

    Here we use SE of sample data, because usually we can't 
    calculate SE for population data, because we don't know the STD of population data.

    -----
    Args:
        a: (list): sample data
        popmean: (float): population mean value
    
    Return:
        statistic (float): t value
        pvalue (float): p value
    """
    sample_mean = mean(a)

    t_val = (sample_mean - popmean) / standard_error(a)
    p_val = t.sf(abs(t_val), len(a)) * 2
    return t_val, p_val


def two_sample_ttest(a: list, b: list) -> Tuple[float, float]:
    """Return T-test statistic value for two independent samples and p value

    Info:
    -----
    We assume that we have 2 independent samples but their means are identical.
    As opposed to pairwise comparison in 'paired T-test', here we compare the means
    of 2 samples, divided by their SE (not really, but it seems like SE).

    If we have just 2 independent samples of data and we want to compare their mean values,
    we can just use 'two_sample_ttest'. If we have data before and after some experiment, 
    then this data is 'dependent' and it's better to use paired ttest.

    Hypothesis:
    -----------
    The null hypothesis (H0): there is no difference between this 2 sample means
        H0: M1 = M2
    The alternative hypothesis (H1): this 2 sample means NOT EQUAL to each other
    and there is a good chances to get some great results
        H1: M1 != M2

    Formula:
    --------
    `t = (X1 - X2) - (M1 - M2) / SQRT( (sd1^2 / n1) + (sd2^2 / n2) )`

    Suppose that M1 - M2 = 0, because we assume in population data M1 and M2 are identical,
    so we can rewrite the formula:

    `t = (X1 - X2) / SQRT( (sd1^2 / n1) + (sd2^2 / n2) )`

    Interpretation:
    ---------------
    This value says, that our difference between 2 sample means deviates 
    from the population mean on "t" sigma value.

    Steps:
    ------
    1. Calculate the mean values for a and b samples
    2. Calculate the STD of a and b samples
    3. In numerator will be difference between a and b means
    4. For denominator we have SQRT of SUM of STD^2 divided by their NO

    -----
    Args:
        a (list): first independent series data
        b (list): second independent series data

    Returns:
        float: statistic value t
        float: pvalue
    """
    X1, X2 = mean(a), mean(b)
    sd1, sd2 = std(a), std(b)
    n1, n2 = len(a) - 1, len(b) - 1

    se1 = sd1 ** 2 / n1
    se2 = sd2 ** 2 / n2

    t_val = (X1 - X2) / (se1 + se2) ** 0.5
    p_val = t.sf(abs(t_val), n1 + n2) * 2

    return t_val, p_val


def paired_ttest(a: list, b: list) -> Tuple[float, float]:
    """Return T Student Criterion statistic value
    
    Info:
    -----
    This criterion allows to compare 2 samples of data
    and called "Paired TTest" or "T-Student Criterion". Paired T-test is used when 
    2 data samples is from one population data.
    Paired samples t-tests are often referred to as "dependent samples t-tests". 
    This is because here we calculate the difference element by element, then calculate
    the mean and standard error of this difference. So this test can be use only, if we
    have some data before experiment and after experiment.

    Usage example:
    --------------
    We can use paired t-test if we are interested in the difference between
    two variables for the same subject. It can be cholesterol level measurement 
    for xxxx year and 1 year later. 

    If we have the group of data before and after experiment, we can use paired T-test.

    Hypothesis:
    -----------
    * H0: The mean pre-test and post-test scores are equal
    * H1: The mean pre-test and post-test scores are not equal

    Formula:
    --------
    `t = Md / SE(Md)`
        where
            Md: The mean difference between a and b samples
            SE: Standard error

    Note:
    -----
    1. To use T Test, it is better that variance from this 2 samples
    were approximately the same (dispersion homogeneity requirement). 
    To check this requirement use "Levene test" or "Fisher criterion".
    2. If NO (number of observations) of sample data is not that much (NO < 30), this is very important
    to have normal distribution of sample data. If NO > 30, this criterion is fine
    even without normal distribution. (But it's steel better to have it tho)

    Steps:
    ------
    1. Calculate the pairwise difference between a and b sample lists (d)
    2. Calculate the mean of the difference (Md)
    3. Calculate the standard error of difference (SE)
    4. Divide Md by SE

    -----
    Args:
        a (list): pre-test data
        b (list): post-test data
    
    Returns:
        float: statistic value t
        float: pvalue
    """
    paired_differences = paired_diff(a, b)
    mean_differences = mean(paired_differences)
    df = len(paired_differences) - 1
    SE = standard_error(paired_differences)
    t_val = mean_differences / SE
    p_val = t.sf(abs(t_val), df) * 2

    return t_val, p_val
