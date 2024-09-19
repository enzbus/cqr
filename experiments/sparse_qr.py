"""Test sparse QR."""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cvxpy as cp

# m = 200
# n = 100

def count_below(sparsity):
    return np.sum(sparsity) - np.sum(np.triu(sparsity))

def row_permute(sparsity, weighter, weighter_mult = 1.):
    row_weight = sparsity @ np.arange(n)[::-1]
    row_weight += (sparsity * weighter).sum(1) * weighter_mult
    row_perm = np.argsort(row_weight)[::-1]
    return sparsity[row_perm]

def col_permute(sparsity, weighter, weighter_mult = 0.):
    col_weight = sparsity.T @ np.arange(m)
    col_weight += (sparsity * weighter).sum(0) * weighter_mult
    col_perm = np.argsort(col_weight)
    return sparsity[:, col_perm]


def sample_problem_matrix(n):
    x = cp.Variable(n)
    objective = cp.Maximize(
        x @ np.random.randn(n) 
        - cp.sum_squares(x @ np.random.randn(n,n)) 
        - cp.norm1(x - np.random.randn(n)))
    constraints = [
        cp.norm1(x) <= 1.,
        cp.norm1(x - np.random.randn(n)) <= 1.,
        x <= np.random.randn(n),
        x >= np.random.randn(n),
        ]
    problem = cp.Problem(objective, constraints)
    # syntax specific for CVXOPT
    return problem.get_problem_data('CVXOPT')[0]['G']

def make_weighter(sparsity):
    big_weighter = np.ones_like(sparsity)
    big_weighter -= np.triu(big_weighter)
    return big_weighter

mat = sample_problem_matrix(100)
# mat = sp.sparse.random(m, n, density=.5, format='csr')
sparsity = (mat.todense().A != 0.) * 1.
print('NONZERO IN matrix', np.sum(sparsity))

w = make_weighter(sparsity)
print(f'HOW MANY BELOW {int(count_below(sparsity))}')
plt.imshow(sparsity)
plt.show()

m, n = mat.shape

for i in range(1):

    # permute rows
    # sparsity = row_permute(sparsity, w, weighter_mult = 0.)
    # print(f'HOW MANY BELOW {int(count_below(sparsity))}')
    # plt.imshow(sparsity)
    # plt.show()

    # permute cols
    sparsity = col_permute(sparsity, w, weighter_mult = 0.)
    print(f'HOW MANY BELOW {int(count_below(sparsity))}')
    # plt.imshow(sparsity)
    # plt.show()

plt.imshow(sparsity)
plt.show()

def pseudo_givens(sparsity, col, high, low):
    """Zero (low,col), set nonzero intersection in cols>=col."""
    for mycol in range(col, n):
        # print('column', mycol)
        if sparsity[high, mycol] == 1:
            # print('element', high, mycol, 'is nonzero')
            sparsity[low, mycol] = 1.
        if sparsity[low, mycol] == 1.:
            # print('element', low, mycol, 'is nonzero')
            sparsity[high, mycol] = 1.
    sparsity[low, col] = 0.

def find_lowermost(sparsity, col):
    subtract = np.argmax(sparsity[:,col][::-1] == 1.)
    return m - subtract - 1

def find_swap_column_above(sparsity, col, lowermost):
    costs = [
        np.sum(sparsity[candidate_row, col:] != sparsity[lowermost, col:]) 
        for candidate_row in range(lowermost)]
    chosen_row = np.argmin(costs)
    assert chosen_row < lowermost
    return chosen_row

TOTAL_GIVENS = 0

for col in range(n):
    print('working on column', col)
    while True:
        lowermost = find_lowermost(sparsity, col)
        if lowermost > col:
            # find which other row to zero it against
            chosen_row = find_swap_column_above(sparsity, col, lowermost)   
            print('zeroing row', lowermost, 'against row', chosen_row)
            pseudo_givens(sparsity, col=col, high=chosen_row, low=lowermost)
            TOTAL_GIVENS += 1
        else:
            print('no more elements below diagonal in column', col)
            # plt.imshow(sparsity)
            # plt.show()
            break

print('TOTAL GIVENS', TOTAL_GIVENS)
print('NONZERO IN R', np.sum(sparsity))

plt.imshow(sparsity)
plt.show()

# curcol = 0
# pseudo_givens(sparsity, curcol, curcol, find_lowermost(curcol))

# pseudo_givens(sparsity, curcol, curcol, find_lowermost(curcol))
# plt.imshow(sparsity)
# plt.show()

# u = np.copy(sparsity[:, 0])
# u[0] = 2
# u /= (np.linalg.norm(u))


# sparsity[:, 0] - ((sparsity[:, 0] @ u) * u) * 2.



# for col in range(5):
#     print('col', col)
#     print('col sparsity below diag', sparsity[col:,col])
