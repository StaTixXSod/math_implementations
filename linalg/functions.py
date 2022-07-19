def mul(a: list, b: list):
    """
    Scalar product of vector "a" and vector "b".

    FORMULA: 
    --------
    ( a, b ) = (a1 * b1) + (a2 * b2) + ... + (an * bn)

    where (a, b) - is a scalar product
    """
    product = 0
    for ai, bi in zip(a, b):
        product += (ai * bi)
    return product

def matmul(A: list, B: list):
    """
    Matrix multiplication

    -----
    INFO:
    -----
    To perform matrix product of matrix A and matrix B, each row vector of matrix A 
    multiplied by the each column vector of matrix B using scalar product (mul).

    Args:
        A (list): Matrix A
        B (list): Matrix B

    Returns:
        list: Matrix product of 2 matrices
    """
    nA, nB = len(A), len(B)
    mA, mB = len(A[0]), len(B[0])

    assert mA == nB

    product = [[0]*mB for _ in range(nA)]
    
    # For each row in A...
    for row_idx in range(nA):
        Ai = A[row_idx]
        # For each col in B...
        for col_idx in range(mB):
            Bi = [B[i][col_idx] for i in range(mA)]
            product[row_idx][col_idx] = mul(Ai, Bi)

    return product

def transpose(matrix: list):
    """Return transposed matrix"""
    transposed_matrix = [list(i) for i in zip(*matrix)]
    return transposed_matrix

def get_vector_norm(v: list):
    """
    Return vector norm (length). Denoted as || v ||.

    Formula:
    --------
    || v || = SQRT(v1^2 + v2^2 + ... + vn^2)
    """
    l = 0
    for i in v:
        l += i**2
    return l**0.5

def identity_matrix_like(n: int) -> list:
    """
    Return Identity matrix with specified shape

    Args: n (int): Matrix shape (n x n)

    Returns: list: Identity matrix
    """
    identity = [[0] * n for _ in range(n)]
    for i in range(n):
        identity[i][i] = 1
    return identity

def split_by_vectors(matrix: list):
    """Splits original data into X and y data by vectors"""
    matrix_by_vector = transpose(matrix=matrix)
    X = matrix_by_vector[:-1]
    y = matrix_by_vector[-1]
    return X, y

def minor(A: list, i: int, j: int):
    """
    Return Minor matrix to calculate determinant of that matrix.

    -----
    INFO:
    -----
    The minor of matrix A, is the same matrix 
    except specified "i" row and "j" column indices. We just get rid of that values.
    Then the determinant of that matrix M is calculated to get Minor value (not in that function).

    --------
    EXAMPLE:
    --------

    Let matrix A be like:

    | a b c |

    | d e f |
    
    | g h i |

    The minor of matrix A of "a" item will be: 

    | e f |
    
    | h i |

    For "e" it will be:

    | a c |
    
    | g i |

    And so on...

    Then, the determinant of that matrix is calculated to get Minor value.

    Args:
        A (list): Original matrix
        i (int): specified row index
        j (int): specified column index

    Returns:
        list: Minor matrix
    """
    M = []
    for row_idx in range(len(A)):
        if row_idx == i:
            continue
        else:
            M.append([])
            for col_idx in range(len(A[0])):
                if col_idx != j:
                    M[-1].append(A[row_idx][col_idx])
    return M
 
def det(A: list) -> float:
    """
    Return matrix determinant (value)

    -----
    INFO:
    -----
    The determinant is the scalar value, calculated from a square matrix (ONLY) (n x n).
    The determinant can help to find out if matrix is invertible. If | A | != 0,
    then the matrix is invertible. Else the inverse of matrix A is doesn't exists.

    --------
    FORMULA:
    --------
    | A | or D(A) : determinant of matrix A (value)

    * Simple method

        If A shape: 2 x 2, like: 

            | a c |

            | b d |

            then: 
            | A | = ad - bc
        
        If A shape: 3 x 3,

            |a1 b1 c1|

            |a2 b2 c2|

            |a3 b3 c3|

            then:
            | A | = a1b2c3 + b1c2a3 + c1a2b3 - a3b2c1 - b3c2a1 - c3a2b1

    * Normal method

    The | A | is equal to sum of products of row elements
    by it's minor value (for first row for example).

    >>> | A | = SUM(
        (+) A[i][j] * M(A[i][j])
        (-) A[i][j+1] * M(A[i][j+1])
        (+) ...
        )

    where 
        i: row index
        j: item index in row
        A: Original matrix
        M: Minor of matrix A with specified row and column 
    
    Args:
        A (list): Original matrix

    Returns:
        float: Determinant value
    """
    m = len(A)
    n = len(A[0])
    if m != n:
        return None
    if n == 1:
        return A[0][0]
    signum = 1
    determinant = 0
 
    for j in range(n):
        determinant += A[0][j] * signum * det(minor(A, 0, j))
        signum *= -1
    return determinant
 
def inverse(A: list):
    """
    Return inverse matrix

    -----
    INFO:
    -----
    Inverse matrix of matrix A is denoted as A^-1. 
    If we make matrix product of A and A^-1, we get I - Identity matrix.
    This is usefull, for example, if we want to find coefficients vector of equation just like:

    >>> Ax = b     | * A^-1
    >>> A @ A^-1 @ x = b @ A^-1
    >>> x = b @ A^-1 (because A @ A^-1 = I)

    --------
    FORMULA:
    --------
    >>> A @ A^-1 = A^-1 @ A = I
    >>> A^-1 = ( 1 / ( | A | ) ) * M.T, 
            
            where: | A | - determinant of A,

                    M.T - transposed M matrix,

                    M - Minor of matrix A

    ------
    STEPS:
    ------
    1. Find determinant of matrix A. 
    If | A | = 0: inverse matrix doesn't exists, because we can't invert the process. 

    2. Find Minor matrix

    3. Switch signs in Minor specific indices

    4. Transpose Minor matrix

    5. Put if all together in formula to find A^-1

    Args:
        A (list): Original matrix

    Returns:
        list: Inverse matrix
    """
    result = [[0] * len(A) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            mi = minor(A, i, j)
            if i + j % 2 == 1:
                result[i][j] = -1 * det(mi) / det(A)
            else:
                result[i][j] = 1 * det(mi) / det(A)
    return result