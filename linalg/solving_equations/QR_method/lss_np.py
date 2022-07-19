import numpy as np

"""
3 3
4 2 1 1
7 8 9 1
9 1 3 2

0.2608695652173913 0.04347826086956526 -0.1304347826086957
"""

n, m = map(int, input().strip().split(" "))

A, b = [], []
for i in range(n):
    row = list(map(float, input().strip().split()))
    A.append(row[:-1])
    b.append(row[-1])

q, r = np.linalg.qr(A)
r_inv = np.linalg.inv(r)
x = r_inv@q.T@b

print("Q: ", q)
print("R: ", r)
print("R inv: ", r_inv)

print(x)