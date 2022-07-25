import os
import sys

sys.path.append(os.getcwd())
from statistics_functions.functions import *
from scipy.stats import t

def simple_ols(x: list, y: list) -> tuple:
    """Return linear regression coefficients using correlation between two samples

    Info:
    -----
    This simple OLS function is used to get coefficient and intercept values using one feature.
    The functions returns next values: intercept, slope, T value and P value.

    Equation:
    ---------
    The intercept and slope allows to construct our formula for future predictions.
    * Intercept is the value, which means what "y" will be equal to, when "x" = 0.
    It is like a bias in our equation.
    * Slope is a slope. This value means the slope of the regression line.
    If slope is positive - it means that with each next "x" value the prediction will
    increase by the value of "slope * X" (and plus bias)

    Regression line formula is: 
    `y = intercept + slope * X`

    This formula is for "one feature" predictions. The more features we have, then 
    the more features will be multiplied and added to equation:

    `y = intercept + slope_1 * X1 + slope_2 * X2 + ... + slope_n * Xn`

    Statistics:
    -----------
    To find out, if our slope is strong enough to make predictions, we can 
    calculate the T and P values for our slope.

    If we imagine, that we are making our experiment multiple times, 
    assuming that there is no relationship in our data, all our 'slopes' must be 
    distributed around value of 0. And now we calculate, what chances are to get 
    this current slope assuming there is no relationship in our data. So if we have 
    p value very low, we can assume, that it is small chance that our current slope 
    is meaningless. So we can decide that our slope is meaningful.

    Just like before we initialize the hypotheses.
    * H0: there is no difference between our slope and 0 (slope is meaningless)
    * H1: our slope has a strong meaning

    P value calculation:
    --------------------
    We are using T distribution here, that almost looks like a normal distribution. The T distribution 
    has a 2 tails and because of that, to calculate the p value for our t value, we must 
    get rid of sign of our t value if it exists and multiply our p value by 2.

    Formula:
    --------
    * `slope = std(y) / std(x) * r_xy`

        where r_xy: correlation between x and y

    * `intercept = Y - slope * X`

        where X and Y: mean values of "x" and "y"

    because:
    `y = intercept + slope * X`   =>    `y - slope * X = intercept`
    
    -----
    Args:
        x (list): Independent values (our feature)
        y (list): Dependent values (data to predict)

    Returns:
        float: Intercept
        float: Slope
        float: T value
        float: P value
    """
    sdy, sdx = std(y), std(x)
    rxy = correlation(x, y)

    slope = (sdy / sdx) * rxy
    intercept = mean(y) - slope * mean(x)

    n1, n2 = len(x)-1, len(y)-1

    se1 = std(x)**2 / n1
    se2 = std(y)**2 / n2
    df = len(x) - 2
    t_val = slope / (se1 + se2)**0.5
    p_val = t.sf(abs(t_val), df)*2

    return (intercept, slope, t_val, p_val)
