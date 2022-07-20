import os
import sys

from QR_functions import normalize_matrix

sys.path.append(os.getcwd())
from linalg.functions import *

def mul(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res

def ortonorm(f):
    n = len(f)
    e = [[] for i in range(n)]
    e[0] = list(f[0])
    trans = [[0] * n for i in range(n)]
    trans[0][0] = 1
    for i in range(1, n):
        e[i] = list(f[i])
        trans[i][i] = 1;
        for j in range(i):
            k = -mul(f[i], e[j]) / mul(e[j], e[j])
            for x in range(m):
                trans[i][x] += k * trans[j][x]
            tmp = list(map(lambda x : x * k, e[j]))
            for x in range(len(f[i])):
                e[i][x] += tmp[x]
    return [e, trans]

n, m = map(int, input().split())
a = [0] * n
e = [[0 for i in range(n)] for i in range(m)]
for i in range(n):
    tmp = list(map(int, input().split()))
    for j in range(m):
        e[j][i] = tmp[j]
    a[i] = tmp[-1]

e, trans = ortonorm(e)
print(e)
e = normalize_matrix(transpose(e))
e = transpose(e)
print(e)
# x = [0] * m
# for i in range(m):
#     for j in range(m):
#         x[j] += mul(e[i], a) / mul(e[i], e[i]) * trans[i][j]
# print(*x)

r_inv = inverse(trans)
b = transpose(list([a]))
C = matmul(transpose(e), b)
x = matmul(r_inv, C)

print(x)