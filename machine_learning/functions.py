import os
import sys

sys.path.append(os.getcwd())
from statistics_functions.functions import *

def make_prediction(x: list, intercept: float, slope: float) -> list:
    """Makes prediction for each "x" value"""
    return [intercept + slope * xi for xi in x]

def residuals(predictions: list, y: list) -> float:
    """Calculate the difference between predictions and actual data using MSE"""
    return mean([(y_hat - yi)**2 for y_hat, yi in zip(predictions, y)])

def sum_of_squares_residuals(predictions: list, y: list) -> float:
    return sum([(yhat - yi)**2 for yhat, yi in zip(predictions, y)])

def sum_of_squares_total(y: list) -> float:
    Y = mean(y)
    return sum([(yi - Y)**2 for yi in y])

def r2_coefficient_score(predictions: list, answers: list) -> float:
    """Return the coefficient of determination

    Info:
    -----
    The coefficient of determination denoted as `R^2`. It shows, 
    how well our predictions fits the data. If the score equals to 1, 
    this means we can totally describe the data with our model.
    If score equals to 0, this means our predictions doesn't better 
    (or actually the same) than the model that just predicts the mean "y" each time. 
    And there is the negative value, that says that our model is worse than "just 
    predict the mean value every time" model.

    Formula:
    --------
    `r^2 = 1 - (ssres / sstotal)`

    where   r^2: coefficient of determination
            ssres: sum of squares for residuals
            sstotal: sum of squares for data and the mean of "y"

    * `ssres`: Sum of squares for residuals is the sum of squared distance between 
    answers and our prediction line (or just predictions)
    * `sstotal`: Sum of squared total is the sum of squared distance between our
    answers and the mean value of our answers.

    Interpretation:
    ---------------
    * If the `ssres` approrimately equals to `sstotal`, this means that the ratio 
    between `ssres` and `sstotal` will be tend to 1. Therefore `r^2 = 1 - 1 -> 0`.
    This means that our model doesn't explain variability of dependent variable.

    * If we have strong positive correlation between X and y, this means our "y" values very 
    close to the regression line and `ssres` will be much less than `sstotal`, because the distance
    from answers to the mean of "y" in the case of strong positive correlation will be much more.
    So the ratio of ssres and sstotal will be tend to 0 and `r^2 = 1 - 0 = 1`.
    This means that approximately 100% of variability of dependent variable (y) explains by the 
    relationship with independent variable (X). 


    ------
    Args:
        predictions (list): prediction list
        answers (list): answer list

    Returns:
        float: r2 score
    """
    ssres = sum_of_squares_residuals(predictions, answers)
    sstotal = sum_of_squares_total(answers)
    r2 = 1 - (ssres / sstotal)
    return r2