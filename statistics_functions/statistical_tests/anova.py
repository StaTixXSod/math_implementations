import sys, os
sys.path.append(os.getcwd())
from typing import Tuple, NamedTuple
from statistics_functions.functions import *
from scipy.stats import f
import pandas as pd

class FactorialAnovaStatistics(NamedTuple):
    sum_sq: float
    df: int
    statistic: float
    pvalue: float

class OneWayAnovaStatistics(NamedTuple):
    statistic: float
    pvalue: float

def one_way_anova(*args: list) -> OneWayAnovaStatistics:
    """
    Return statistic value of the "one way anova" and it's pvalue

    Info:
    -----
    ANOVA: Analysis of Variance.

    One way anova compare all passed groups to each other and:
    * H0 says: there is no difference between this groups (M1 = M2 = ... = Mn)
    * H1 says: at least one group of data has different sample mean

    We assume that the variance of our data can be explained by two things:
    * Variability between the groups
    * Variability within the groups

    If SSB much greater than SSW, then probably at least 2 mean values are different from each other.
    To find out, how likely is that, we calculate the p-value, using F value.

    Interpretation:
    ---------------
    Lets assume, we repeating our experiment many times when H0 is TRUE.
    It means, that all our sample means ARE EQUAL. It's like we getting 3 samples from 1 population.
    So, our SSB tends to 0, because we assume, if we have H0, there is no difference between sample means.
    SSW is the total value of difference within the group, it's like the correction to our computation.
    This means, that in most cases, if H0 is TRUE, the F value will be very small.

    Here we have F distribution. This distribution not like the normal distribution, 
    it has asymmetry, skewed to the left (positive skew) (Sometimes max value is close to 0). If we get large F value, we may reject the null hypothesis.
    To reject the null hypothesis, we calculate the p-value.

    Steps:
    ------
    1. Find sum of squares between groups
    2. Find sum of squares within groups
    3. Calculate the "F" value

    Formula:
    --------
    `F = (SSB / df_ssb) / (SSW / df_ssw)`

    -----
    Args:
    sample_1, sample_2, ... ,sample_n: (list) Samples of data to compare

    Return:
    statistic: (float) statistic F value
    pvalue: (float) pvalue for statistic
    """
    ssb, df_ssb = SSB(args)
    ssw, df_ssw = SSW(args)

    ssb_div = ssb / df_ssb
    ssw_div = ssw / df_ssw

    f_val = ssb_div / ssw_div
    p_val = f.sf(f_val, df_ssb, df_ssw)
    
    return OneWayAnovaStatistics(f_val, p_val)

def TSS(vector: list) -> float:
    """
    Return total sum of squares

    Info:
    -----
    This value shows, how high is the variability of our data, without dividing data into groups.
    It's like variance, but we are not going to divide by "NO".

    Formula:
    --------
    `TSS = SUM( (yi - Y)^2 )`
        
        where
            yi: item in vector,
            Y: vector mean value

    `df = NO - 1`

        where
            df: Degrees of freedom
            NO: Number observations

    Steps:
    ------
    1. Calculate one general mean of all groups
    2. For each item in vector calculate the squared difference between this item and the general mean
    3. Sum this difference values

    -----
    Args:
        vector (list): 1D vector.

    -------
    Return:
        tss: float
            The calculated total sum of squares
        df: int
            The degrees of freedom
    """
    Y = mean(vector)
    return sum([(yi - Y)**2 for yi in vector])

def SSB(groups: list) -> Tuple[float, int]:
    """Return the sum of squares between groups and ddof

    Info:
    -----
    The SSB (sum of squares between groups) means, how in total 
    the mean of group (m) deviates from its general mean (X).

    If there is a big difference between mean in group and the general mean, 
    we can assume, that there is an difference between groups and we 
    may reject the null hypothesis. Otherwise if SSB is a small value, then we
    may assume, that there is not much difference between mean in groups. 

    Formula:
    --------
    `SSB = NO * (m - X)`

        where
            NO: Number observations
            m: Mean within the group
            X: General mean of all groups

    `df = number_of_groups - 1`

    -----
    Args:
        groups (list): Groups of our data observations

    Return:
        ssb: float
            The calculated sum of squares between the groups
        df: int
            The degrees of freedom
    """
    X = mean(flatten(groups))
    df = len(groups) - 1
    ssb = sum([len(group) * (mean(group) - X)**2 for group in groups])
    return (ssb, df)

def SSW(groups: list) -> Tuple[float, int]:
    """Return the sum of squares within groups and ddof

    Info:
    -----
    The SSW (sum of squares within groups) shows, how much 
    an items in the groups deviates from its mean in group.
    If SSW is a small value, then the data within the groups doesn't deviates much
    from it's mean. Otherwise there is big variance in the groups.

    If SSW is small and SSB is large, there is a chance to get rid of the H0 hypothesis.

    Formula:
    --------
    `SSW = SUM(TSS(group_i))`

        where
            TSS: Total sum of squares for each group
            group_i: Specific group

    `df = SUM(NO_i - 1)`
        
        where
            NO_i: Number observations in group

    -----
    Args:
        groups (list): Groups of our data observations

    Return:
        ssw: float
            The calculated sum of squares within the groups
        df: int
            The degrees of freedom
    """
    df = sum([len(group)-1 for group in groups])
    ssw = sum([TSS(group) for group in groups])
    return (ssw, df)

def two_way_anova(data: pd.DataFrame, features: list, target: str) -> FactorialAnovaStatistics:
    """Return the f statistic of data and its pvalue

    Info:
    -----
    A two-way ANOVA (“analysis of variance”) is used to determine whether
    or not there is a statistically significant difference between the means 
    of three or more independent groups that have been split on two variables 
    (sometimes called “factors”).

    Args:
        data (pd.DataFrame): DataFrame with data to compare
        features (list): the features to compare

    Returns:
        Tuple[float, float]: F statistic and p value
    """
    feature_data = {}
    groupped = data.groupby(features)
    for group, df in groupped:
        name = "_".join([str(g) for g in group])
        feature_data[name] = df[target].values
        # feature_data[group] = df[target].values
    feature_data = pd.DataFrame(feature_data)

    mean_values = feature_data.mean()
    std_values = feature_data.std()

    print(pd.DataFrame({
        "N": feature_data.shape[0],
        "Mean": mean_values,
        "STD": std_values
    }))

    mean_by_feature = {}
    for feature in features:
        feature_group = data.groupby(feature)
        for group, df in feature_group:
            mean_by_feature[group] = [np.mean(df[target])]

    mean_by_feature = pd.DataFrame(mean_by_feature)
    grand_mean = np.mean(mean_values)
    ss_first_feature = np.sum(np.square(mean_by_feature - grand_mean), axis=1)

    # sum_sq = ...
    # df = ...
    # t_val = ...
    # p_val = ...


def SST(data: np.array, grand_mean):
    """Return Sum of squares total

    Args:
        vector (pd.DataFrame): data
    """
    X = np.mean(data)
    sst = [(xi - X)**2 for xi in data]

