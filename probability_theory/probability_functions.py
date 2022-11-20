from fractions import Fraction


def intersection_probability(na: int, nb: int, nobs: int) -> Fraction:
    """
    Return intersection probability between A and B
    `A ∩ B`: interaction between A and B
    """
    pa = Fraction(na, nobs)
    pb = Fraction(nb, nobs)

    return pa * pb


def union_probability(na: int, nb: int, nobs: int, dependent: bool = True) -> Fraction:
    """
    Return union probability for two sets.
    If this 2 events may occur together, it means these events are dependent.

    - if we just count all possible probabilities, that cannot occur simultaneously, we just `sum it up`
    - if events are dependent and we need to find the P of (A OR B), we should sum it up and subtract their intersection

    `A ∪ B`: interaction between A and B
    """
    pa = Fraction(na, nobs)
    pb = Fraction(nb, nobs)
    if dependent:
        pab = intersection_probability(na, nb, nobs)
        return pa + pb - pab
    return pa + pb


def only_a_probability(na: int, nb: int, nobs: int) -> Fraction:
    """Return A\B (A and not B) probability"""
    intersection = intersection_probability(na, nb, nobs)
    pa = Fraction(na, nobs)
    return pa - intersection


def only_b_probability(na: int, nb: int, nobs: int) -> Fraction:
    """Return B\A (B and not A) probability"""
    intersection = intersection_probability(na, nb, nobs)
    pb = Fraction(nb, nobs)
    return pb - intersection
