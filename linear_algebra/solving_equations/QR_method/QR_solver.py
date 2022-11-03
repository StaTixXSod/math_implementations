from QR_functions import *

n, m = map(int, input().strip().split(" "))

matrix = []
for i in range(n):
    matrix.append(list(map(float, input().strip().split(" "))))

coefs = solve_with_qr_decomposition(matrix)
print(*coefs)

"""
4 2
4 2 8
5 2 4
2 6 2
3 0 8

1.6531 -0.30894
"""

"""
3 3
4 2 1 1
7 8 9 1
9 1 3 2

0.2608695652173913 0.04347826086956526 -0.1304347826086957
"""

"""
3 1
4 1
7 1
9 2

0.19863013698630136
"""