#
# Trick to get solver object
#
import numpy as np
import logging
logging.basicConfig(level='INFO')
import cvxpy as cp
from cqr.cvxpy_interface import CQR

SOC = False # True #False #True #True #False
PO = False # False #True
PROGRAM_ONE = False#False#True # True #False
PROGRAM_TWO = True # True # False # True #True #True



if SOC:
    np.random.seed(0)
    m, n = 2000, 1000
    x = cp.Variable(n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    objective = cp.norm2(A @ x - b) + 1. * cp.norm1(x)
    constraints = []
    program = cp.Problem(cp.Minimize(objective), constraints)
    #prog.solve(solver='SCS', verbose=True, eps=3e-10)
    #scs_obj = objective.value
    program.solve(solver=CQR())

if PO:
    np.random.seed(0)
    n = 50
    w = cp.Variable(n)
    w0 = np.random.randn(n)
    # w0 = np.maximum(w0,0.)
    w0 -= np.sum(w0)/len(w0)
    w0 /= np.sum(np.abs(w0))
    mu = np.random.randn(n) * 1e-3
    Sigma = np.random.randn(n,n)
    Sigma = Sigma.T @ Sigma
    eival, eivec = np.linalg.eigh(Sigma)
    eival *= 1e-4
    eival[:-len(eival)//10] = 0.
    Sigma = eivec @ np.diag(eival) @ eivec.T
    # chol = np.linalg.cholesky(Sigma)
    # objective = cp.quad_form(w, Sigma) + w.T @ mu + 1e-4 * cp.norm1(w-w0)
    objective = w.T @ mu + 1e-5 * cp.norm1(w-w0)
    # w_max = np.random.randn(n)
    # w_max = np.maximum(w_max,0.01)
    # w_max = np.minimum(w_max,0.05)
    constraints = [#w >=0, #w<=w_max,
     cp.sum(w)==0, cp.norm1(w-w0)<=0.05, 
        cp.norm1(w)<=1, cp.sum_squares((np.diag(np.sqrt(eival)) @ eivec.T) @ w) <= 0.00005]
    program = cp.Problem(cp.Minimize(objective), constraints)
    #prog.solve(solver='SCS', verbose=True, eps=3e-10)
    #scs_obj = objective.value
    program.solve(solver=CQR())
    # breakpoint()

if PROGRAM_ONE:
    seed=0
    m=401
    n=300
    np.random.seed(seed)
    x = cp.Variable(n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    objective = cp.norm1(A @ x - b)
    d = np.random.randn(n, 50)
    constraints = [cp.abs(x) <= .75, x @ d == 2.,]
    program = cp.Problem(cp.Minimize(objective), constraints)
    program.solve(solver=CQR())


if PROGRAM_TWO:
    seed=0
    m=70
    n=40
    x = cp.Variable(n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    objective = cp.norm1(A @ x - b) + 1. * cp.norm1(x)
    # adding these constraints, which are inactive at opt,
    # cause cg loop to stop early
    constraints = []  # x <= 1., x >= -1]
    program = cp.Problem(cp.Minimize(objective), constraints)
    program.solve(solver=CQR())


###
# Simple DR
###


from cqr.cvxpy_interface import solvers
self = solvers[-1]


dr_y = np.zeros(2 * self.m)
losses = []
xs = []
steps = []
for i in range(5000):
    xs.append(np.copy(dr_y))
    step = self.douglas_rachford_step(dr_y)
    steps.append(np.copy(step))
    losses.append(np.linalg.norm(step))
    print(losses[-1])
    if losses[-1] < 1e-10:
        print(f'converged in {i} iterations')
        break
    dr_y += step
else:
    pass #raise Exception

var_reduced = self._var_reduced_from_sy(dr_y)
print('SQNORM RESIDUAL OF SOLUTION',
    np.linalg.norm(self.newres(var_reduced))**2)


import matplotlib.pyplot as plt
# plt.semilogy(losses)
# plt.show()

oldlosses = np.array(losses)


import scipy as sp

BROYDEN_TYPE = 2

for MEMORY in [1,2,5,10,20,50]:
    print()
    print('MEMORY', MEMORY)
    import cvxpy as cp
    mystep = np.array(steps[-MEMORY-1:])
    Y = np.diff(mystep, axis=0)
    myxs = np.array(xs[-MEMORY-1:])
    S = np.diff(myxs, axis=0)
    if BROYDEN_TYPE == 2:
        Jinv = cp.Variable((self.m*2, self.m*2))
        J0inv = -np.eye(self.m*2)
        objective = cp.Minimize(
            cp.sum_squares((Jinv-J0inv) / (np.linalg.norm(J0inv)**2))
            + cp.sum_squares((Jinv @ Y.T + S.T)/(np.linalg.norm(S)**2)))
        cp.Problem(objective).solve()#verbose=True)
        print('TYPE 2')
        print('no ', np.linalg.norm(self.douglas_rachford_step(xs[-1] + steps[-1])))
        print('fwd', np.linalg.norm(self.douglas_rachford_step(xs[-1] + Jinv.value @ steps[-1])))
        print('bwd', np.linalg.norm(self.douglas_rachford_step(xs[-1] - Jinv.value @ steps[-1])))


        ## NEW
        S = -S
        S_scaled = S * (np.sqrt(self.m * 2) / np.linalg.norm(S))**2
        Y_scaled = Y * (np.sqrt(self.m * 2) / np.linalg.norm(S))**2
        C = Y_scaled.T
        D = S_scaled.T
        u,s,v = np.linalg.svd(C, full_matrices=False)
        newJ = (-np.eye(self.m*2) + D @ C.T) @ (np.eye(self.m*2) - u @ np.diag(s**2 / (1 + s**2)) @ u.T)
        print('newfwd', np.linalg.norm(self.douglas_rachford_step(xs[-1] + newJ @ steps[-1])))
        print('newbwd', np.linalg.norm(self.douglas_rachford_step(xs[-1] - newJ @ steps[-1])))


    if BROYDEN_TYPE == 1: # DOES TERRIBLE!!!!
        J = cp.Variable((self.m*2, self.m*2))
        J0 = -np.eye(self.m*2)
        objective = cp.Minimize(
            cp.sum_squares((J-J0) / (np.linalg.norm(J0)**2))
            + cp.sum_squares((J @ S.T + Y.T)/(np.linalg.norm(S)**2)))
        cp.Problem(objective).solve()#verbose=True)
        print('TYPE 1')
        print('no ', np.linalg.norm(self.douglas_rachford_step(xs[-1] + steps[-1])))
        print('fwd', np.linalg.norm(self.douglas_rachford_step(xs[-1] + np.linalg.solve(J.value, steps[-1]))))
        print('bwd', np.linalg.norm(self.douglas_rachford_step(xs[-1] - np.linalg.solve(J.value, steps[-1]))))



def _densify(linear_operator):
    """Create Numpy 2-d array from a sparse LinearOperator."""
    result = np.zeros(linear_operator.shape, dtype=float)
    for j in range(result.shape[1]):
        ej = np.zeros(result.shape[1])
        ej[j] = 1.
        result[:, j] = linear_operator @ ej
    return result

zio = _densify(self.newjacobian_linop(self.var_reduced))


raise Exception


for i in range(10):
    xs.append(np.copy(dr_y))
    step = self.douglas_rachford_step(dr_y)
    steps.append(np.copy(step))
    losses.append(np.linalg.norm(step))
    print(i, losses[-1])
    if losses[-1] < 1e-13:
        print(f'converged in {i} iterations')
        break
    test_new_x = {}


    # if i % 27 == 0:
    #     dr_y = self._sy_from_var_reduced(self.inexact_levemberg_marquardt(self.refinement_residual, self.refinement_jacobian, self._var_reduced_from_sy(dr_y), max_iter=20))
    #     continue

    result = sp.sparse.linalg.lsmr(-self.douglas_rachford_step_derivative(dr_y), steps[-1], atol=0., btol=0.)#, iter_lim=30)
    dr_y = np.copy(xs[-1] + result[0]/10)

    oldnorm = np.linalg.norm(steps[-1])
    for i in range(10):
        testnorm = np.linalg.norm(self.douglas_rachford_step(xs[-1] +  result[0]*(0.5)**i))
        if testnorm < oldnorm:
            dr_y = np.copy(xs[-1] + result[0]*(0.5)**i)
            break
    else:
        print('FAILED!')
        dr_y = np.copy(xs[-1] + steps[-1])
    continue
    breakpoint()

    MEMORY = 2

    if MEMORY == 0:
        dr_y = np.copy(xs[-1] + steps[-1])#@*0.15)
        continue
    else:
        mystep = np.array(steps[-MEMORY-1:])
        Y = np.diff(mystep, axis=0)
        myxs = np.array(xs[-MEMORY-1:])
        S = np.diff(myxs, axis=0)

        newstep = np.copy(steps[-1])
        result = np.zeros_like(newstep)
        for index in range(MEMORY):
            norm = np.linalg.norm(Y[-1-index])
            proj = (newstep/norm) @ (Y[-1-index]/norm)
            result -= S[-1-index] * proj
            newstep -= Y[-1-index] * proj
        
        newstep *= .5
        newstep += result

        dr_y = np.copy(xs[-1] + newstep)

        # oldnorm = np.linalg.norm(steps[-1])
        # for i in range(1):
        #     testnorm = np.linalg.norm(self.douglas_rachford_step(xs[-1] + newstep*(0.5)**i))
        #     if testnorm < oldnorm:
        #         dr_y = np.copy(xs[-1] + newstep*(0.5)**i)
        #         break
        # else:
        #     print('FAILED!')
        #     dr_y = np.copy(xs[-1] + steps[-1])



    #     test_new_x[MEMORY] = np.copy(xs[-1] + newstep)#@*0.15)
    
    # # test_new_x[MEMORY] = zio
    # # test_new_x[MEMORY] = newzio
    # # test_new_x[MEMORY] = newnewzio

    #     # breakpoint()

    # test_new_losses = {MEMORY:np.linalg.norm(self.douglas_rachford_step(test_new_x[MEMORY])) for MEMORY in test_new_x}

    # BEST_MEMORY = np.argmin(list(test_new_losses.values()))
    # #print('BEST MEMORY', list(test_new_losses.keys())[BEST_MEMORY])
    # dr_y = test_new_x[list(test_new_losses.keys())[BEST_MEMORY]]

self.var_reduced=self._var_reduced_from_sy(dr_y)
# self.refine()
# self.new_toy_solve()
# self.decide_solution_or_certificate()

plt.semilogy(losses)
plt.semilogy(oldlosses)
plt.show()

