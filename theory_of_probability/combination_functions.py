from fractions import Fraction
from numbers import Real


def factorial(n: int) -> int:
    """
    Returns the factorial of a number
    Formula:
    n! = ∏(ni),
        where:
            `∏`: cumulative product of all numbers
            ni: each value in the list
    """
    f = 1
    for num in range(2, n+1):
        f *= num
    return f


def cumprod(a: list) -> Real:
    """Return the cumulative product"""
    assert len(a) > 1
    c = a[0]
    for num in a[1:]:
        c *= num
    return c


def P(n: int) -> Real:
    """
    The permutation function. Returns the number of permutations for the specified number.

    Permutation without repetitions:
    --------------------------------
    The permutation without repetitions is the simple factorial of n. It means, how many permutations we can make
    with specified number of objects. We use the factorial here, because in the first time we have n options to take
    some object. Then we have n - 1 options, because don't return our objects in basket. Then n - 2 and so on.
    The permutation without repetitions formula is just the product of `n * n-1 * n-2 * ... * n-n+1` or just `n!`.

    Args:
        n: the number of options

    Returns: the number of permutation for specified number of options
    """
    return factorial(n)


def C(n: int, m: int, repetition: bool = False) -> Real:
    """
    Returns the number of combinations for the specified number of observations (n) and size of each combination (m).
    Formula:
        C without repetition = n! / ((n - m)! * m!)
        C with repetition = (n + m - 1)! / ((n - 1)! * m!)
    Args:
        n: (int) number of observations (options)
        m: (int) the size of each combination
        repetition: (bool) if True - with repetition, else - without repetition

    Returns: the number of combinations
    """
    if repetition:
        return factorial(n + m - 1) / (factorial(n - 1) * factorial(m))
    else:
        return factorial(n) / (factorial(n - m) * factorial(m))


def A(n: int, m: int, repetition: bool = False) -> Real:
    """
    Returns the number of accommodations for the specified number of observations (n) and size of each combination (m).
    Args:
        n: (int) number of observations (options)
        m: (int) the size of each combination
        repetition: (bool) if True - with repetition, else - without repetition

    Returns: the number of accommodations
    """
    if repetition:
        return n**m
    else:
        return factorial(n) / factorial(n - m)
