from solve_with_QR import *

n, m = map(int, input().strip().split(" "))

matrix = []
for i in range(n):
    matrix.append(list(map(float, input().strip().split(" "))))

coefs = solve_linear_equation_QR(matrix)
print(*coefs)

"""
4 2
4 2 8
5 2 4
2 6 2
3 0 8

1.6531 -0.30894
"""
