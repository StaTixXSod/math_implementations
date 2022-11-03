from fractions import Fraction


def independent_intersection_probability(na: int, nb: int, nobs: int) -> Fraction:
    """
    Return intersection probability between A and B
    `A ∩ B`: interaction between A and B
    """
    pa = Fraction(na, nobs)
    pb = Fraction(nb, nobs)
    return pa * pb


def union_probability(na: int, nb: int, nobs: int) -> Fraction:
    """
    Return union of two sets of probability
    `A ∪ B`: interaction between A and B
    """
    pa = Fraction(na, nobs)
    pb = Fraction(nb, nobs)
    return pa + pb


def only_a_probability(na: int, nb: int, nobs: int) -> Fraction:
    """Return A\B (A and not B) probability"""
    intersection = independent_intersection_probability(na, nb, nobs)
    pa = Fraction(na, nobs)
    return pa - intersection


def only_b_probability(na: int, nb: int, nobs: int) -> Fraction:
    """Return B\A (B and not A) probability"""
    intersection = independent_intersection_probability(na, nb, nobs)
    pb = Fraction(nb, nobs)
    return pb - intersection
