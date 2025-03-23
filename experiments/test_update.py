import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
N = 100
MEMORY = 5

np.random.seed(0)
# mystep = np.array(steps[-MEMORY-1:])
# Y = np.diff(mystep, axis=0)
# myxs = np.array(xs[-MEMORY-1:])
# S = np.diff(myxs, axis=0) # CAREFUL - sign

Y = np.random.randn(MEMORY, N)*1e-7
S = np.random.randn(MEMORY, N)*1e-7


# PROG 0
#if BROYDEN_TYPE == 2:
Jinv = cp.Variable((N, N))
J0inv = -np.eye(N)
objective = cp.Minimize(
    cp.sum_squares((Jinv-J0inv)) / (np.linalg.norm(J0inv)**2)
    + cp.sum_squares((Jinv @ Y.T - S.T))/(np.linalg.norm(S)**2))
cp.Problem(objective).solve()#verbose=True)
sol0 = Jinv.value

# PROG 1
objective = cp.Minimize(
    cp.sum_squares((Jinv-J0inv) / (np.linalg.norm(J0inv)))
    + cp.sum_squares((Jinv @ Y.T - S.T)/(np.linalg.norm(S))))
cp.Problem(objective).solve()#verbose=True)
sol1 = Jinv.value

# PROG 2
S_scaled = S * (np.sqrt(N) / np.linalg.norm(S))
Y_scaled = Y * (np.sqrt(N) / np.linalg.norm(S))

objective = cp.Minimize(
    cp.sum_squares(Jinv-J0inv)
    + cp.sum_squares(Jinv @ Y_scaled.T - S_scaled.T))
cp.Problem(objective).solve()#verbose=True)
sol2 = Jinv.value

plt.plot(np.linalg.svd(sol0)[1])
plt.plot(np.linalg.svd(sol1)[1])
plt.plot(np.linalg.svd(sol2)[1])

plt.show()

B = J0inv
C = Y_scaled.T
D = S_scaled.T

zio = (B + D @ C.T) @ np.linalg.inv(np.eye(N) + C @ C.T)
zia = -np.eye(N) + D @ C.T - (-C + D @ C.T @ C) @ np.linalg.inv(np.eye(MEMORY) + C.T @ C) @ C.T

plt.plot(np.linalg.svd(sol0)[1])
plt.plot(np.linalg.svd(zio)[1])
plt.plot(np.linalg.svd(zia)[1])

u,s,v = np.linalg.svd(C, full_matrices=False)
# C = u @ np.diag(s) @ v
# C.T = v.T @ np.diag(s) @ u.T

zietta = -np.eye(N) + D @ C.T - (-np.eye(N) + D @ C.T) @ (u @ np.diag(s**2 / (1 + s**2)) @ u.T)

zietto = (-np.eye(N) + D @ v.T @ np.diag(s) @ u.T) @ (np.eye(N) - u @ np.diag(s**2 / (1 + s**2)) @ u.T)
zione = D @ v.T @ np.diag(s / (1 + s**2)) @ u.T - (np.eye(N) - u @ np.diag(s**2 / (1 + s**2)) @ u.T)


assert np.allclose(zietto, sol0)
assert np.allclose(zione, sol0)


plt.plot(np.linalg.svd(zietta)[1])
plt.plot(np.linalg.svd(zietto)[1])
plt.plot(np.linalg.svd(zione)[1])


plt.show()


# PROG 2

# Tr((Jinv - J0inv).T @ (Jinv - J0inv) + (Jinv @ Y_scaled.T - S_scaled.T).T @ (Jinv @ Y_scaled.T - S_scaled.T))
# Tr(Jinv.T @ Jinv - Jinv - Jinv.T + Y_scaled @ Jinv.T @ Jinv @ Y_scaled.T - S_scaled @ Jinv @ Y_scaled.T - Y_scaled @ Jinv.T @ S_scaled.T)
# Tr(Jinv.T @ Jinv) - 2*Tr(Jinv) + Tr(Y_scaled @ Jinv.T @ Jinv @ Y_scaled.T) - 2*Tr(S_scaled @ Jinv @ Y_scaled.T)

# 2 * Jinv - 2 * I + 2 * Y_scaled @ Jinv @ Y_scaled.T - 2*S_scaled  @ Y_scaled.T
# Jinv - I + Y_scaled @ Jinv @ Y_scaled.T - S_scaled  @ Y_scaled.T
# Jinv + Y_scaled @ Jinv @ Y_scaled.T = I + S_scaled  @ Y_scaled.T
