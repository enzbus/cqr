import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tqdm

from cqr.ql_transform import data_ql_transform, backward_transform_ql


np.random.seed(0)
m = 2000
n = 300
# A = np.random.randn(m,n)
A = sp.sparse.random(m,n,density=.01).todense().A
x = np.random.randn(n)
z = np.random.randn(m)
y = np.maximum(z, 0.)
s = y - z
c = -(A.T @ y)
b = A @ x + s


# test solve with SCS
import cvxpy as cp
x = cp.Variable(n)
constraint = [A @ x -b <= 0]
problem = cp.Problem(cp.Minimize(x.T @ c), constraint)
problem.solve(
    # solver='OSQP', verbose=True, eps_abs = 1.0e-12, eps_rel = 1.0e-12,
    solver='SCS', verbose=True, eps=1e-14, # acceleration_lookback=0,
)

Q_orig = np.block([
    [np.zeros((n,n)), A.T , c.reshape(n,1)],
    [-A, np.zeros((m,m)), b.reshape(m,1)],
    [-c.reshape(1,n), -b.reshape(1,m), np.zeros((1,1))]
])
assert np.allclose(Q_orig, -Q_orig.T)


# ANALYZE SCS SOLUTION
x_scs = problem.solver_stats.extra_stats['x']
y_scs = problem.solver_stats.extra_stats['y']
s_scs = problem.solver_stats.extra_stats['s']
u_scs = np.concatenate([x_scs, y_scs, [1.]])
v_scs = np.concatenate([np.zeros(n), s_scs, [0.]])
print(
    'SCS FINAL LOSS',
    np.linalg.norm(Q_orig @ u_scs - v_scs)
    )

# WE USE QR TRANSF ONLY FOR CQR PROTOTYPE

NEWQR = True
SCALE = False
if NEWQR:
    A, c, b, (q_orth, scale), L = data_ql_transform(A, b, c)
    print('ORIG SCALE', scale)
    if not SCALE:
        scale = 1.
    A /= scale
    b /= scale
    c /= scale
else:
    scale = 1.
    L = np.eye(n+1)

Q = np.block([
    [np.zeros((n,n)), A.T , c.reshape(n,1)],
    [-A, np.zeros((m,m)), b.reshape(m,1)],
    [-c.reshape(1,n), -b.reshape(1,m), np.zeros((1,1))]
])
assert np.allclose(Q, -Q.T)


# projection
u = np.random.randn(n+m+1)
v = np.random.randn(n+m+1)

# # projection with CVXPY
# import cvxpy as cp
# def project_cvxpy(u0, v0):
#     u = cp.Variable(n+m+1)
#     v = cp.Variable(n+m+1)
#     objective = cp.Minimize(cp.sum_squares(u - u0) + cp.sum_squares(v - v0))
#     cp.Problem(objective, [Q @ u == v]).solve()
#     return u.value, v.value
# u_cp, v_cp = project_cvxpy(u, v)

# # with numpy
# def project(u0, v0):
#     u = np.linalg.solve(np.eye(n+m+1) + Q.T @ Q, u0 + Q.T @ v0)
#     return u, Q @ u
# u1, v1 = project(u, v)
# assert np.allclose(u_cp, u1)
# assert np.allclose(v_cp, v1)

# with scipy
MAT = np.eye(n+m+1) + Q.T @ Q
# plt.plot(1./np.linalg.eigh(MAT)[0])
# plt.show()

LU_FACTOR = sp.linalg.lu_factor( np.eye(n+m+1) + Q.T @ Q)
def project_sp(u0, v0):
    u = sp.linalg.lu_solve(LU_FACTOR, u0 + Q.T @ v0)
    return u, Q @ u
u2, v2 = project_sp(u, v)
# assert np.allclose(u_cp, u2)
# assert np.allclose(v_cp, v2)

# conic part
def project_cones(u0, v0):
    u = np.copy(u0)
    u[n:] = np.maximum(u[n:], 0.)
    v = np.zeros_like(v0)
    v[n:] = np.maximum(v0[n:], 0.)
    return u, v

# DR splitting
losses = []
lens = []
for i in tqdm.tqdm(range(20000)):
    u_pi_cone, v_pi_cone = project_cones(u, v)
    losses.append(np.linalg.norm(Q @ u_pi_cone - v_pi_cone))
    u_pi_aff, v_pi_aff = project_sp(2 * u_pi_cone - u, 2 * v_pi_cone - v)
    step_len = np.sqrt(
        np.linalg.norm(u_pi_aff - u_pi_cone)**2 +
        np.linalg.norm(v_pi_aff - v_pi_cone))
    lens.append(step_len)
    u += (u_pi_aff - u_pi_cone)*1.0
    v += (v_pi_aff - v_pi_cone)*1.0
u, v = project_cones(u, v)

# apply scale
v *= scale

# transform back to get loss
u1_orig, tau_orig, v1_orig, kappa_orig = backward_transform_ql(
    u[:n], u[-1], v[:n], v[-1], n, L)
u_orig = np.concatenate([u1_orig, u[n:-1], [tau_orig]])
v_orig = np.concatenate([v1_orig, v[n:-1], [kappa_orig]])
print(
    'CQR PROTOTYPE FINAL LOSS',
    np.linalg.norm(Q_orig @ u_orig - v_orig)
    )




plt.semilogy(losses)
plt.semilogy(lens)
plt.show()