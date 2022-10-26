from typing import Tuple
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use("fivethirtyeight")
plt.rcParams['figure.figsize'] = (12, 8)


def pearson_distance_between_two_groups(o1: int, o2: int) -> float:
    """
    Returns Pearson's chi-squared distance.

    Info:
    -----
    Suppose we have observations for categorical data. We don't have some histogram or smth.
    And we have to find out if the data of our categorical observations differ from each other.
    The easiest way is to subtract observed value from expected, square and sum them up to get rid of negative values
    and find the whole distance. The problem here is, that no matter what number of observations we have, the distance
    will remain the same. So we need to normalize our data by dividing this value by expected value.

    Steps:
    ------
    1. Find expected values
    2. Find the squared difference between observed and expected values
    3. Normalize them by expected group value
    4. Sum the difference to get Pearson Chi-squared distance between 2 groups

    Formula:
    --------
    X^2 = Σ(Oi - Ei)^2 / Ei,
        where:
            Oi: observed group value
            Ei: expected group value

    Args:
        o1 (int): observed value for the first group
        o2 (int): observed value for the second group

    Returns:
        difference (float): the calculated difference between groups
    """
    # 1. Find expected values
    nobs = o1 + o2
    e1 = e2 = nobs // 2
    # 2. Find the difference between observed and expected values
    diff1 = (o1 - e1) ** 2 / e1
    diff2 = (o2 - e2) ** 2 / e2
    diff = diff1 + diff2
    return diff


def chi_squared_distance(observed: np.ndarray, expected: np.ndarray = None) -> float:
    if expected is None:
        expected = np.mean(observed)
    return ((observed - expected) ** 2 / expected).sum()


def chi2_test(observed: np.ndarray, expected: np.ndarray = None) -> Tuple[float, float]:
    """
    Perform Chi Squared test for observed data frequencies. If expected is None, it'll be the mean of observed data
    Returns:
        Chi statistic (float)
        P value (float)
    """
    chi_statistic = chi_squared_distance(observed, expected)
    p_value = chi2.sf(chi_statistic, len(observed) - 1)
    return chi_statistic, p_value


def simulate_chi2_distribution_for_coin_tosses(n_toss: int = 60, n_tests: int = 1000):
    """
    Simulate coin tosses and plot Chi Square distribution.

    Each observed group should have the normal distribution. The thing is here, that if we have 1 observed group (
    actually 2 groups, but with 1 degree of freedom), our mean will be around 0, so the most of all values for Chi
    distribution will be around 0 (because of square we have positive values only). But, as the number of observed
    groups increases, the small values, that distributed around the 0 will be accumulated and Chi square distribution
    will tend to a normal distribution.
    """

    def _toss_coin(n_attempts: int) -> Tuple[int, int]:
        """Toss coin n times and return 2 observed values"""
        attempts = np.random.randint(2, size=n_attempts)  # array([1, 0, 0, 0, 1, 1, ..., 0, 0, 1, 0])
        observed2 = sum(attempts)
        observed1 = n_attempts - observed2
        return observed1, observed2

    tosses = {
        "eagle": [],
        "tails": [],
        "distances": []
    }

    for i in range(n_tests):
        o1_rand, o2_rand = _toss_coin(n_toss)
        d = pearson_distance_between_two_groups(o1_rand, o2_rand)
        tosses['eagle'].append(o1_rand)
        tosses['tails'].append(o2_rand)
        tosses['distances'].append(d)

    bins = 15
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.hist(tosses['eagle'], bins=bins)
    ax2.hist(tosses['tails'], bins=bins)
    ax3.hist(tosses['distances'], bins=bins)

    ax1.set_title("Eagle distribution")
    ax2.set_title("Tails distribution")
    ax3.set_title("Chi2 distribution")

    plt.tight_layout()
    plt.show()
    return


def simulate_chi_distribution_with_n_degrees(max_ddof: int = 10):
    """
    Chi Square Distribution is the SUM of Squares of independent random standard values (the sum of normally distributed
    values). The more groups we have, the more chances we'll get our Chi distribution be like normal distribution,
    because each normal distribution, that have 0 as mean and 1 as std will be accumulated, so the sum of all small
    values will make our Chi distribution be like normal.

    Args:
        max_ddof: The maximum value for degrees of freedom

    Returns:
        plot Chi Square distribution for specified range of DDOFs
    """
    x = np.linspace(0, 10, 100)
    for ddof in range(1, max_ddof, 2):
        plt.plot(x, chi2.pdf(x, ddof), label=f'k={ddof}')

    plt.title("Chi Distribution with different degrees of freedom")
    plt.legend(bbox_to_anchor=(1, 1), fontsize=14)
    plt.show()
    return


if __name__ == "__main__":
    # n1 = np.array([10, 10, 10, 5, 10, 15])
    n1 = np.array([10, 30, 50])
    print(chi2_test(n1))
