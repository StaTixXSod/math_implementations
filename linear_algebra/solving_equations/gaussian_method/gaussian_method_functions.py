import os
import sys
sys.path.append(os.getcwd())

from typing import List
from linear_algebra import functions


class EquationError(Exception):
    """Can't solve equation"""


matrix_shape = List[List[float]]
vector_shape = List[float]


def is_solvable(matrix: matrix_shape) -> str:
    """
    To find, if system has solutions or not, usually use determinant.
    But here, due of using Gaussian method it can be calculated with this method:

    - if in any row, ALL values is equals to 0, then the system either has INF solutions, or has no solutions at all
    - so if all values in row equals to 0 and corresponding answer is equal to 0 too -> then the system has
    "INFINITE" solutions - otherwise if corresponding answer not equals to 0 -> then the system HAS NO solutions -
    Else we have 1 solution

    Args:
        matrix (list of lists): Triangular matrix, calculated in the previous step

    Returns:
        str: The answer if system of linear equations has solutions: "YES" / "NO" / "INF"

    ----------
    Examples:
    ----------
    -----------------
    Has one solution:
    -----------------
    9.0  1.0  3.0  2.0

    0.0  7.22  6.66  -0.55

    0.0  0.0  -1.77  0.23

    Returns: YES

    -----------------------
    Has infinite solutions:
    -----------------------
    5.0  -14.0  15.0  0.0
    
    0.0  6.4  -8.0  0.0
    
    0.0  0.0  0.0  0.0
    
    INF

    -----------------
    Has no solutions:
    -----------------
    6.0  8.0  -7.0  2.0
    
    0.0  -7.33  10.66  -1.33
    
    0.0  0.0  0.0  1.0 (here only zeros except answer value)
    
    0.0  0.0  -4.0  -2.0
    
    NO
    """
    for row in matrix:
        a, b = row[:-1], row[-1]
        if all(round(v, 2) == 0 for v in a):
            if b == 0:
                return "INF"
            else:
                return "NO"

    return "YES"


def sort_matrix(matrix: matrix_shape, ascending: bool = False) -> matrix_shape:
    """Sort matrix by first element in rows"""
    if ascending:
        return sorted(matrix, key=lambda x: x[0])
    else:
        return sorted(matrix, key=lambda x: x[0], reverse=True)


def combine_matrices(X: matrix_shape, y: vector_shape) -> matrix_shape:
    """Return full matrix"""
    matrix = []

    for X_row, y_num in zip(X, y):
        full_row = [i for i in X_row]
        full_row.append(y_num)
        matrix.append(full_row)

    return matrix


def forward_pass(X: matrix_shape) -> matrix_shape:
    """
    Forward pass is the method, that allows to transform original matrix to its triangular form.

    Args:
        X (matrix_shape): Original system matrix
    Returns:
        matrix_shape: Triangular matrix

    -------------------------------------

    Steps:
    ------
    1. Combine system matrix and answer vector to get a full matrix,
    that will be further transformed into a triangular matrix

    2. Sort rows in descending order by first element in row to make
    the largest coefficients be on the first place

    3. For each row (step) proceed function 'forward_step'

    Example:
    ---------
    The goal is to get this kind of matrix:

    x1 x2 x3 | b

    0  x2 x3 | b

    0  0  x3 | b
    """
    triangular_matrix = sort_matrix(X)

    n = len(X) if len(X) < len(X[0]) else len(X[0])

    for step in range(n):
        triangular_matrix = forward_step(tr_m=triangular_matrix, step=step)

    return triangular_matrix


def forward_step(tr_m: matrix_shape, step: int) -> matrix_shape:
    """
    This is one step from forward pass function.
    Here function nullifies first coefficient element of a certain row/column
    and based on that multiplicative value recalculate the rest values from that row. 

    Args:
        `tr_m` (matrix_shape): Triangular matrix. Matrix, that transforms to triangular here
        `step` (int): Current step (row number)

    ---------------------
    Steps:
    ---------------------
    1. Find the main value (coefficient), which will be multiplied and subtracted for nullification value below

    2. If that main value is 0, then do nothing and continue step due that you can't divide on 0

    3. `For` loop from step + 1 to n:
        - find the value to nullify and its multiplication (mult) value
        - initialize `for` loop over column indices
        - change the rest values in row by subtracting from certain value specific number (value above * mult value)

    ---------------------
    Before row transform:
    ---------------------
    x1 x2 x3 | b

    x1 x2 x3 | b
    
    --------------------
    After row transform:
    --------------------
    x1 x2 x3 | b

    0  x2 x3 | b

    In addition, x2 and x3 from 2nd equation subtracted by value above, multiplied by a coefficient

    """
    n = len(tr_m)  # Number of rows
    m = len(tr_m[0])  # Number of columns
    coefficient = tr_m[step][step]  # Current coefficient

    if coefficient == 0:  # We can't divide on 0, so continue here
        return tr_m

    for row_idx in range(step + 1, n):
        nullify_val = tr_m[row_idx][step]
        mult = -nullify_val / coefficient

        for col_idx in range(m):
            num_above = tr_m[step][col_idx]
            curr_number = tr_m[row_idx][col_idx]

            tr_m[row_idx][col_idx] = curr_number + (num_above * mult)

    return tr_m


def backward_pass(matrix: matrix_shape) -> vector_shape:
    """
    This backward pass function finds the coefficients of equation, using triangular matrix.

    Args:
        matrix (matrix_shape): Triangular matrix

    Returns:
        vector_shape: unknown variables to solve equation

    ------------------------------------------------
    Steps:
    -------
    1. Create coefficients list to have access to specific coefficient by its index in future

    2. Initialize `for` loop in reverse order from 'number of rows' to 0 due to the fact,
    that we "unsqueeze" our equation from bottom to top

    3. Define next values:
        - b: these are the values from the last column called a "constant"
        - known variables: it's the sum of (coefficient * parameter), that non zero
        - answer: when we know some coefficients, we can find some variables and put that values to the right
        - value: it's the coefficient of unknown variable
        - Coefficient is the answer / value
    """
    n = len(matrix)
    coefficients: vector_shape = [0 for _ in range(n)]

    for row_idx in range(n - 1, -1, -1):
        b = matrix[row_idx][-1]
        known_variables = sum([coefficients[i] * matrix[row_idx][i] for i in range(n)])
        answer = b - known_variables
        value = matrix[row_idx][row_idx]
        coefficients[row_idx] = answer / value

    return coefficients


def gaussian_solve_equation(X: matrix_shape, y: vector_shape, show_triangular: bool = False) -> vector_shape:
    """
    This function returns a coefficients of the given matrix using Gaussian method.

    Args:
        X (matrix_shape): Original system matrix
        y (vector_shape): Constant vector
        show_triangular (bool): If `True`, then show the matrix

    Returns:
        vector_shape: Coefficients to solve equation

    ------------------------------------
    Steps:
    -------
    1. Proceed forward pass

    2. Decide, if matrix is solvable

    3. If 'YES', proceed backward pass
    """
    if len(X) < len(X[0]):
        raise EquationError("The equation has infinite solutions...")

    matrix = combine_matrices(X, y)
    triangular_matrix = forward_pass(matrix)
    solvable = is_solvable(triangular_matrix)

    if show_triangular:
        functions.print_matrix(triangular_matrix, title="Triangular matrix", use_round=4)

    if solvable != "YES":
        raise EquationError(f"The equation has {solvable} solutions...")

    else:
        return backward_pass(triangular_matrix)
