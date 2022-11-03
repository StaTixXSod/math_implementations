import os
import sys

sys.path.append(os.getcwd())
from typing import Tuple, NamedTuple

import pandas as pd
from scipy.stats import f
from statistics.functions import *


class OneWayAnovaStatistics(NamedTuple):
    statistic: float
    pvalue: float


class GroupStatistic(NamedTuple):
    sum_sq: float or list
    df: int or list
    ms: float or list


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
    1. Find the sum of squares between groups
    2. Find the sum of squares within groups
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
    `TSS = Σ( (yi - Y)^2 )`
        
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
    Y = np.mean(vector)
    return sum([(yi - Y) ** 2 for yi in vector])


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
    ssb = sum([len(group) * (mean(group) - X) ** 2 for group in groups])
    return ssb, df


def SSW(groups: list) -> Tuple[float, int]:
    """Return the sum of squares within groups and ddof

    Info:
    -----
    The SSW (sum of squares within groups) shows, how much 
    an items in the groups deviates from its mean in group.
    If SSW is a small value, then the data within the groups doesn't deviate much
    from its mean. Otherwise, there is big variance in the groups.

    If SSW is small and SSB is large, there is a chance to get rid of the H0 hypothesis.

    Formula:
    --------
    `SSW = Σ(TSS(group_i))`

        where
            TSS: Total sum of squares for each group
            group_i: Specific group

    `df = Σ(NO_i - 1)`
        
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
    df = sum([len(group) - 1 for group in groups])
    ssw = sum([TSS(group) for group in groups])
    return ssw, df


def two_way_anova(data: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    """Return two-way ANOVA result

    Info:
    -----
    Two-way ANOVA is almost the same as one-way ANOVA. The difference
    here is, that for two-way ANOVA we trying to compare all features data with all groups 
    in all features data. This allows to find out, what feature has the most impact on the data.

    The process pretty the same, as in the One-way ANOVA: find SSB, SSW and calculate the ratio between this 2 values.

    Steps:
    ------
    1. Calculate the sum of squares between groups
    2. Calculate the sum of squares within groups
    3. Get total sum of squares
    4. Calculate interaction of features
    5. Calculate F values
    6. Calculate P values
    7. Form the dataframe with statistic

    -----
    Args:
        data (pd.DataFrame): DataFrame with data to compare
        features (list): the features to compare
        target: the target feature we want to explore

    Returns:
        Pandas DataFrame: Statistics about the data
    """
    # 1. Find the sum of squares for each feature (SSB: Sum of squares between groups)
    ssb = SSB_Factorial(data, features, target)

    # 2. Find the sum of squares within groups SSW (residuals)
    ssw = SSW_Factorial(data, features, target)

    # 3. Calculate total sum of squares
    ss_total = TSS(data[target].values)

    # 4. Calculate Sum of Squares Interaction
    interactions = interaction(ss_total, ssb, ssw)

    # 5. Calculate F values
    f_values = calculate_F_values(ssb, ssw, interactions)

    # 6. Calculate P values
    p_values = calculate_P_values(f_values, ssb, ssw)

    # 7. Group it all together
    statistic = form_statistic(ssb, ssw, interactions, f_values, p_values, features)

    return statistic


def SSB_Factorial(data: pd.DataFrame, features: list, target: str) -> GroupStatistic:
    """Return sum of squares, degrees of freedom and mean sum of squares between groups

    Info:
    -----
    The SSB is a metric, that shows, how much the difference between our groups is. 
    The Factorial SSB means, that we calculate the means for each feature we want to compare.
    We then compare all the means with the general mean by squaring and summing all the values up.
    So, we want to find out how much the mean of specific feature differs from the general mean.

    Formula:
    --------
    * `sum_sq = Σ(ni(fm - GM)**2)`

    where:
        `sum_sq`: sum of squares
        `ni`: number of elements in specific group
        `fm`: feature mean
        `GM`: grand mean

    * `df = nf - 1`

    where:
        `df`: degrees of freedom
        `nf`: number of features

    * `ms = sum_sq / df`

    where:
        `ms`: mean sum of squares

    -----
    Args:
        data (pd.DataFrame): data
        features (list): feature list
        target (str): dependent variable

    Returns:
        GroupStatistic: group statistic
    """
    features_sum_sq = []
    features_df = []
    features_ms = []
    GM = np.mean(data[target].values)  # Grand Mean

    # For each feature...
    for feature in features:
        feature_means = []  # add up the means for each group of feature
        feature_group = data.groupby(feature)  # group by feature
        # For each group with feature...
        for _, feature_df in feature_group:
            values = feature_df[target]
            n = values.shape[0]
            feature_mean = np.mean(values)
            feature_means.append((n, feature_mean))

        sum_sq = np.sum([n * (fm - GM) ** 2 for n, fm in feature_means])
        df = data[feature].nunique() - 1
        features_sum_sq.append(sum_sq)
        features_df.append(df)
        features_ms.append(sum_sq / df)

    return GroupStatistic(features_sum_sq, features_df, features_ms)


def SSW_Factorial(data: pd.DataFrame, features: list, target: str) -> GroupStatistic:
    """Return sum of squares, degrees of freedom and mean sum of squares within groups

    Info:
    -----
    The process within groups means, that we split our data by features and groups in features.
    For each group we calculate the same squared sum and trying to find out, how much the values
    of group differs from the mean of group. If sum of squares of this differences not too much,
    but SSB is big, we can assume we have strong difference in some feature.

    Formula:
    --------
    * `sum_sq = Σ(ss)`
    * `ss = Σ((vi - M)**2)`

    where:
        `sum_sq`: sum of squares
        `ss`: sum of squares for specific group
        `vi`: value in group
        `M`: group mean

    * `df = Σ(nfi - 1)`

    where:
        `df`: degrees of freedom
        `nfi`: number of features for specific group

    * `ms = sum_sq / df`

    where:
        `ms`: mean sum of squares

    -----
    Args:
        data (pd.DataFrame): data
        features (list): feature list
        target (str): dependent variable

    Returns:
        GroupStatistic: group statistic
    """
    groupped = data.groupby(features)
    ss_resid_list = []
    df_resid = 0
    for _, groupped_df in groupped:
        values = groupped_df[target]
        group_mean = np.mean(values)
        ss = np.sum([(vi - group_mean) ** 2 for vi in values])
        ss_resid_list.append(ss)
        df_resid += groupped_df.shape[0] - 1
    ss_resid = np.sum(ss_resid_list)
    ms_resid = ss_resid / df_resid

    return GroupStatistic(ss_resid, df_resid, ms_resid)


def interaction(total: float, ssb: GroupStatistic, ssw: GroupStatistic) -> GroupStatistic:
    """Return feature interaction statistic

    Info:
    -----
    The interaction means, how much variability explains by feature interaction.
    To calculate the feature interaction we must subtract from total sum of squares the whole pack of calculations:
    SSB for each feature and SSW (residuals). 
    
    If our SSB is low, SSW is low, 
    but TTS is high, this means, that our variability of data may be explained by the interaction 
    of our features. 

    Formula:
    --------
    * `A ∩ B = TTS - SSB - SSW`

    where:
        `A ∩ B`: interaction between feature A and feature B
        `TTS`: total sum of squares
        `SSB`: sum of squares between groups
        `SSW`: sum of squares within groups

    * `df = ∏(ssb_df)`

    where
        `∏`: cumulative product of all degrees of freedom for each feature
        `ssb_dt`: degrees of freedom for each feature

    -----
    Args:
        total (float): total sum of squares
        ssb (GroupStatistic): sum of squares between groups
        ssw (GroupStatistic): sum of squares within groups

    Returns:
        GroupStatistic: interaction sum_sq, df and ms
    """
    ss = total - np.sum(ssb.sum_sq) - ssw.sum_sq
    df = np.cumprod(ssb.df)[-1]
    ms_interaction = ss / df
    return GroupStatistic(ss, df, ms_interaction)


def calculate_F_values(ssb: GroupStatistic, ssw: GroupStatistic, interactions: GroupStatistic) -> list:
    """Calculate the F values

    Info:
    -----
    To calculate the statistic value for our features, we need to divide each 'sum of squares' feature 
    value between groups (SSB) onto the sum of squares within groups (SSW). The same process is valid for 
    the 'interaction' sum_sq values. We can't get the negative value, because sum of squares is a 
    strictly positive number, therefore here we have "F" distribution. Because if we don't have 
    significant differences, then the mean of F values will be distributed around the 1 or something.

    So, if our sum of squares between groups (SSB) is much larger than the sum of squares within groups (SSW), 
    we can assume we have strong statistical significance. But this has to be verified with the pvalue calculation.

    Formula:
    --------
    `f_value = MSSB / MSSW`

    where:
        `MSSB`: mean sum of squares between groups
        `MSSW`: mean sum of squares within groups

    -----
    Args:
        ssb (GroupStatistic): SSB statistic
        ssw (GroupStatistic): SSW statistic
        interactions (GroupStatistic): interaction statistic

    Returns:
        list: F values list
    """
    f_features = [ssb_ms_item / ssw.ms for ssb_ms_item in ssb.ms]
    f_interaction = interactions.ms / ssw.ms
    return [*f_features, f_interaction]


def calculate_P_values(f_values: list, ssb: GroupStatistic, ssw: GroupStatistic) -> list:
    """Calculate the P value for each feature and interaction F values

    Info:
    -----
    To calculate P value of specific F statistic use f.sf method. 
    "F" distribution is one tailed, so we don't need to multiply the value by 2 and passing 
    absolute value of statistic, just like in case of "T" distribution.

    To calculate P value of specific statistic, pass to f.sf method "F" statistic, 
    degrees of freedom between groups (SSB.df) and degrees of freedom within groups (SSW.df).

    The "sf" stands for Survival function and calculated as: (1 - cdf) at x of the given RV.
    * RV is a random variables pack of specific distribution
    * cdf is a cumulative distribution function, that return cumulative probability 
    up to the specific ("F" value here) value

    So, if we subtract the cdf from 1, we will get the cumulative probability of the chances, 
    that our result has statistically significant differences. This is what called pvalue and
    if the pvalue <= 0.05, we can assume we have significant sample differences.

    Formula:
    --------
    * `p = f.sf(f_value, ssb, ssw)`
    * `sf = 1 - cdf(f_value, ssb, ssw)`

    -----
    Args:
        f_values (list): F values list
        ssb (GroupStatistic): SSB statistic
        ssw (GroupStatistic): SSW statistic

    Returns:
        list: P values list
    """
    return [f.sf(f_value, ssb.df, ssw.df)[-1] for f_value in f_values]


def form_statistic(
        ssb: GroupStatistic,
        ssw: GroupStatistic,
        interactions: GroupStatistic,
        f_values: list,
        p_values: list,
        features: list) -> pd.DataFrame:
    statistic = {"feature": [], "sum_sq": [], "df": [], "F": [], "Pvalue": []}

    # For each feature fill in corresponded values...
    for sq, df, feature in zip(ssb.sum_sq, ssb.df, features):
        statistic['feature'].append(feature)
        statistic['sum_sq'].append(sq)
        statistic['df'].append(df)

    # Fill in values for interaction...
    statistic["feature"].append(":".join([*features]))
    statistic["sum_sq"].append(interactions.sum_sq)
    statistic["df"].append(interactions.df)

    # Fill in residuals values...
    statistic["feature"].append("residual")
    statistic["sum_sq"].append(ssw.sum_sq)
    statistic['df'].append(ssw.df)

    # Fill F and P values...
    for f_val, p_val in zip(f_values, p_values):
        statistic["F"].append(f_val)
        statistic["Pvalue"].append(p_val)

    statistic = pd.DataFrame.from_dict(statistic, orient='index').T
    return statistic
