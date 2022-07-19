from gaussian_method_functions import *

n, m = map(int, input().split(" "))

X = []
y = []

for i in range(n):
    equation_line = input().strip().split(" ")
    Xi = list(map(float, equation_line[:-1]))
    yi = list(map(float, equation_line[-1:]))
    X.append(Xi)
    y.append(yi[0])

"""
3 3
4 2 1 1
7 8 9 1
9 1 3 2

YES
0.2608695652173913 0.04347826086956526 -0.1304347826086957
"""

# Coefficients: 
y_hat = gaussian_solve_equation(X, y)

if y_hat is not None:
    print(*y_hat)
