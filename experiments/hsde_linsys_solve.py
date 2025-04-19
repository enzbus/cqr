# solve by HSDE system

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tqdm

from cqr.equilibrate import hsde_ruiz_equilibration

TEST_QR = True
EQUILIBRATE = False

np.random.seed(0)
m = 100
n = 30
A = np.random.randn(m,n)
x = np.random.randn(n)
z = np.random.randn(m)
y = np.maximum(z, 0.)
s = y - z
c = -(A.T @ y)
b = A @ x + s

Ac = np.vstack([A, c.reshape(1,n)])
q, r = np.linalg.qr(Ac)
newAc = q
A = q[:-1]
c = q[-1]

Q_orig = np.block([
    [np.zeros((n,n)), A.T , c.reshape(n,1)],
    [-A, np.zeros((m,m)), b.reshape(m,1)],
    [-c.reshape(1,n), -b.reshape(1,m), np.zeros((1,1))]
])
assert np.allclose(Q_orig, -Q_orig.T)
MAT_orig = np.eye(n+m+1) + Q_orig.T @ Q_orig

# structured
low_right = np.block([
    [np.zeros((m,m)), b.reshape(m,1)],
    [-b.reshape(1,m), np.zeros((1,1))]
])

Q = np.block([
    [np.zeros((n,n)), newAc.T],
    [-newAc, low_right]
])

# Qt = np.block([
#     [np.zeros((n,n)), -newAc.T],
#     [newAc, low_right.T]
# ])

assert np.allclose(Q, -Q.T)
assert np.allclose(Q, Q_orig)

MAT = np.block([
    [2 * np.eye(n), -newAc.T @ low_right],
    [-low_right.T @ newAc, np.eye(m+1) + newAc @ newAc.T + low_right.T @ low_right]
])
part_one = newAc.T @ low_right
part_two = low_right.T @ low_right

assert np.allclose(part_one.T @ part_one, low_right.T @ (newAc @ newAc.T) @ low_right)

assert np.allclose(MAT, MAT_orig)

MAT = np.block([
    [2 * np.eye(n), -part_one],
    [-part_one.T, np.eye(m+1) + newAc @ newAc.T + part_two]
])

assert np.allclose(MAT, MAT_orig)

y = np.random.randn(n+m+1)
y1 = y[:n]
y2 = y[n:]
# solve MAT @ x = y
x = np.linalg.solve(MAT, y)
x1 = x[:n]
x2 = x[n:]

# 2 * x1 - part_one @ x2 = y1
# -part_one.T @ x1 + x2 + newAc @ newAc.T @ x2 + part_two @ x2 = y2
#
# x1 = y1/2 + part_one @ x2/2
# -part_one.T @ (y1/2 + part_one @ x2/2) + x2 + newAc @ newAc.T @ x2 + part_two @ x2 = y2

# (I + newAc @ newAc.T + part_two - part_one.T @ part_one / 2) @ x2 = 
#   y2 + part_one.T @ y1 / 2

test1 = ( # WE NEED TO SOLVE BY THIS MATRIX
    np.eye(m+1)
    + newAc @ newAc.T 
    + low_right.T @ low_right 
    - low_right.T @ (newAc @ newAc.T)/2. @ low_right) @ x2
test2 = y2 + low_right.T @ newAc @ y1 / 2

assert np.allclose(test1, test2)

# break it up (part_two)
test = low_right.T @ low_right
test3 = np.block([
    [np.outer(b,b), np.zeros((m,1))],
    [np.zeros((1, m)), b.T @ b]
])
assert np.allclose(test, test3)

# break it up (part_one)
test = newAc.T @ low_right
# low_right = np.block([
#     [np.zeros((m,m)), b.reshape(m,1)],
#     [-b.reshape(1,m), np.zeros((1,1))]
# ])
test4 = np.block([
    [-np.outer(newAc[-1],b), (newAc[:-1].T @ b).reshape(n,1)],
])
assert np.allclose(test, test4)

test5 = test4.T @ test4
tmp = newAc[:-1].T @ b
test6 = np.block([
    [newAc[-1].T @ newAc[-1] * np.outer(b,b), -(b * (newAc[-1]@tmp)).reshape(m,1)],
    [-(b * (newAc[-1]@tmp)).reshape(1,m), tmp.T @ tmp]
])
assert np.allclose(test6, test5)

# PUT IT TOGETHER
# using only newAc and b
newA = newAc[:-1]
newc = newAc[-1]
Atb = newA.T @ b # it is zero with new QR
SOLVE_MATRIX = (
    np.eye(m+1)
    + newAc @ newAc.T 
    + np.block([
        [np.outer(b,b)* (1 - c.T @ c / 2), +(b * (c@Atb)/2.).reshape(m,1)],
        [+(b * (c@Atb)/2.).reshape(1,m), b.T @ b - (Atb.T @ Atb)/2.]
    ])
)

OLD = (
    np.eye(m+1)
    + newAc @ newAc.T 
    + low_right.T @ low_right 
    - low_right.T @ (newAc @ newAc.T)/2. @ low_right)
assert np.allclose(SOLVE_MATRIX, OLD)


