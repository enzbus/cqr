# BSD 3-Clause License

# Copyright (c) 2024-, Enzo Busseti

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Solver main function, will be unpacked and call all the rest."""

import time

import numpy as np
import scipy as sp

NOHSDE = True

from project_euromir import equilibrate
from project_euromir.lbfgs import minimize_lbfgs

if not NOHSDE:
    from project_euromir.loss import (_densify, common_computation_main,
                                      create_workspace_main, gradient, hessian,
                                      loss)
else:
    from project_euromir.loss_no_hsde import (create_workspace,  loss_gradient, hessian, _densify)

from project_euromir.newton_cg import _epsilon, _minimize_newtoncg
from project_euromir.refinement import refine

DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt

QR_PRESOLVE = False
QR_PRESOLVE_AFTER = False
REFINE = True

CG_REGULARIZER = 0.

SCIPY_EXPERIMENTS = False

HESSIAN_COUNTER = {'ba': 0}

def solve(matrix, b, c, zero, nonneg, lbfgs_memory=10):
    "Main function."

    print(
        f'PROBLEM SIZE: m={len(b)}, n={len(c)}, zero={zero},'
        f' nonneg={nonneg}, nnz(matrix)={matrix.nnz}')

    # some shape checking
    n = len(c)
    m = len(b)
    assert matrix.shape == (m, n)
    assert zero + nonneg == m

    if QR_PRESOLVE:
        q, r = np.linalg.qr(np.vstack([matrix.todense(), c.reshape((1, n))]))
        matrix_transf = q[:-1].A
        c_transf = q[-1].A1
        sigma = np.linalg.norm(b) / np.mean(np.linalg.norm(matrix_transf, axis=1))
        b_transf = b/sigma

    else:
        # equilibration
        d, e, sigma, rho, matrix_transf, b_transf, c_transf = \
        equilibrate.hsde_ruiz_equilibration(
                matrix, b, c, dimensions={
                    'zero': zero, 'nonneg': nonneg, 'second_order': ()},
                max_iters=25)
        if QR_PRESOLVE_AFTER:
            q, r = np.linalg.qr(np.vstack(
                [matrix_transf.todense(), c_transf.reshape((1, n))]))
            matrix_transf = q[:-1].A
            c_transf = q[-1].A1

    if False: # test for woodbury approach
        breakpoint()
        import matplotlib.pyplot as plt
        y_part_base = matrix_transf @ matrix_transf.T
        # y_part = y_part.todense()
        mask = (np.random.uniform(size=m) > 0 ) * 1.
        mask[:zero] = 0.
        y_part = y_part_base + np.diag(mask)
        rand = np.random.randn(m)
        nors = []
        nors1 = []
        nors2 = []
        for i in range(100):
            sol = sp.sparse.linalg.cg(y_part, rand, rtol=0., maxiter=i)[0]
            sol1 = sp.sparse.linalg.lsqr(y_part, rand, atol=0., btol=0., iter_lim=i)[0]
            sol2 = sp.sparse.linalg.lsmr(y_part, rand, atol=0., btol=0., maxiter=i)[0]
            nors.append(np.linalg.norm(y_part @ sol - rand)**2)
            nors1.append(np.linalg.norm(y_part @ sol1 - rand)**2)
            nors2.append(np.linalg.norm(y_part @ sol2 - rand)**2)
        sol1 = np.linalg.lstsq(y_part, rand, rcond=None)[0]
        nors.append(np.linalg.norm(y_part @ sol1 - rand)**2)
        plt.plot(nors)
        plt.plot(nors1)
        plt.plot(nors2)
        print(nors[-1], nors[-2])
        plt.yscale('log')

        rand += np.random.randn(m) * 1e-8
        # mask[7:8] = 1 - mask[7:8]
        y_part = y_part_base + np.diag(mask)
        start_point = np.copy(sol)
        nors = []
        nors1 = []
        nors2 = []
        for i in range(50):
            sol = sp.sparse.linalg.cg(y_part, rand, x0=np.copy(start_point), rtol=0., maxiter=i)[0]
            nors.append(np.linalg.norm(y_part @ sol - rand)**2)
            sol1 = sp.sparse.linalg.lsqr(y_part, rand, x0=np.copy(start_point), atol=0., btol=0., iter_lim=i)[0]
            sol2 = sp.sparse.linalg.lsmr(y_part, rand, x0=np.copy(start_point), atol=0., btol=0., maxiter=i)[0]
            nors.append(np.linalg.norm(y_part @ sol - rand)**2)
            nors1.append(np.linalg.norm(y_part @ sol1 - rand)**2)
            nors2.append(np.linalg.norm(y_part @ sol2 - rand)**2)
        plt.figure()
        plt.plot(nors)
        plt.plot(nors1)
        plt.plot(nors2)
        plt.yscale('log')
        plt.show()
        breakpoint()

        raise Exception

    # temporary, build sparse Q
    Q = sp.sparse.bmat([
        [None, matrix_transf.T, c_transf.reshape(n, 1)],
        [-matrix_transf, None, b_transf.reshape(m, 1)],
        [-c_transf.reshape(1, n), -b_transf.reshape(1, m), None],
        ]).tocsc()

    # breakpoint()

    if not NOHSDE:

        # prepare workspace
        workspace = create_workspace_main(Q, n, zero, nonneg)

        # these functions should be unpacked inside newton-cg
        def separated_loss(u):
            common_computation_main(u, Q, n, zero, nonneg, workspace)
            return loss(u, Q, n, zero, nonneg, workspace)

        def separated_grad(u):
            common_computation_main(u, Q, n, zero, nonneg, workspace)
            return np.copy(gradient(u, Q, n, zero, nonneg, workspace))

        def separated_hessian(u):
            common_computation_main(u, Q, n, zero, nonneg, workspace)
            return hessian(u, Q, n, zero, nonneg, workspace)

        x_0 = np.zeros(n+m+1)
        x_0[-1] = 1.

    else:

        workspace = create_workspace(m, n, zero)

        # these functions should be unpacked inside newton-cg
        def separated_loss(xy):
            return loss_gradient(
                xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf, c=c_transf, workspace=workspace)[0]

        def separated_grad(xy):
            return np.copy(loss_gradient(
                xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf, c=c_transf, workspace=workspace)[1])

        def separated_hessian(xy):
            return hessian(
                xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf, c=c_transf, workspace=workspace, regularizer=CG_REGULARIZER)

        def dense_hessian(xy):
            return _densify(hessian(
                xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf, c=c_transf, workspace=workspace, regularizer=1e-10))

        def hessp(xy, var):
            # breakpoint()
            HESSIAN_COUNTER['ba'] += 1
            H = separated_hessian(xy)
            return H @ var

        x_0 = np.zeros(n+m)

    old_x = np.empty_like(x_0)

    def callback(current):
        print('current loss', current.fun)
        # print('current kappa', current.x[-1])
        # if current.fun < np.finfo(float).eps**2:
        #     raise StopIteration
        # if current.x[-1] < 1e-2:
        #     current.x[:] = old_x
        #     raise StopIteration
        #     # current['x'] /= current.x[-1]
        # old_x[:] = current.x

    # import matplotlib.pyplot as plt
    # all_losses = []
    # def callback_plot(x):
    #     all_losses.append(separated_loss(x))

    start = time.time()
    # call newton-CG, implementation from scipy with modified termination
    c_2 = 0.1
    for i in range(5):
        if not SCIPY_EXPERIMENTS:
            result = _minimize_newtoncg(
                fun=separated_loss,
                x0=x_0,
                args=(),
                jac=separated_grad,
                hess=separated_hessian,
                hessp=None,
                # callback=callback,
                xtol=0., #1e-5,
                eps=_epsilon, # unused
                maxiter=10000,
                # max_cg_iters=100,
                disp=99,
                return_all=False,
                c1=1e-4, c2=c_2)
            print(result)
        else:
            # NO, trust region (at least scipy implementations) don't work as
            # well, I tried a bunch, only realistic competitor is l-bfgs
            #global HESSIAN_COUNTER
            HESSIAN_COUNTER['ba'] = 0
            result = sp.optimize.minimize(
                fun=separated_loss,
                x0=x_0,
                args=(),
                jac=separated_grad,
                # hess=dense_hessian, # for dogleg
                #callback=lambda x: print(separated_loss(x)),
                callback=callback_plot,
                hessp=hessp, # for trust-ncg
                options = {
                'initial_trust_radius': .01,
                'max_trust_radius': .1,
                'eta': 0., 'gtol': 1e-16,
                'ftol': 0.,
                'gtol': 0.,
                'maxfun': 1e10,
                'maxiter': 1e10,
                'maxls': 100,
                'maxcor': 20,
                },
                # method='dogleg',
                # method='trust-ncg',
                method='l-bfgs-b',
                )
            print("NUMBER HESSIAN MATMULS", HESSIAN_COUNTER['ba'])
            print(result)
        # plt.plot(all_losses)
        # plt.yscale('log')
        # plt.show()
        # breakpoint()
        print(f'NEWTON-CG took {time.time() - start:.3f} seconds')
        loss_after_cg = separated_loss(result["x"])
        print(f'LOSS {loss_after_cg:.2e}')
        if loss_after_cg < 1e-25:
            break
        x_0 = result['x']
        # breakpoint()
        while separated_loss(x_0 - separated_grad(x_0)) < separated_loss(x_0):
            x_0 -= separated_grad(x_0)

    if not NOHSDE:
    # extract result
        u = result['x']
        assert u[-1] > 0, 'Certificates not yet implemented'
    else:
        xy = result['x']
        u = np.empty(n+m+1, dtype=float)
        u[:-1] = xy
        u[-1] = 1.

    # u /= u[-1]
    v = Q @ u

    if REFINE:
        for _ in range(5): #n + m):
            u, v = refine(
                z=u-v, matrix=matrix_transf, b=b_transf, c=c_transf, zero=zero,
                nonneg=nonneg)#, max_iters=30)

    # Transform back into problem format
    u1, u2, u3 = u[:n], u[n:n+m], u[-1]
    v2, v3 = v[n:n+m], v[-1]

    if v3 > u3:
        raise NotImplementedError('Certificates not yet implemented.')

    # Apply HSDE scaling
    x = u1 / u3
    y = u2 / u3
    s = v2 / u3

    if QR_PRESOLVE:
        x_orig = np.linalg.solve(r, x) * sigma
        y_orig = y
        s_orig = s * sigma

    else:
        if QR_PRESOLVE_AFTER:
            x = np.linalg.solve(r, x)
        # invert Ruiz scaling, copied from other repo
        x_orig =  e * x / sigma
        y_orig = d * y / rho
        s_orig = (s/d) / sigma

    u = np.concatenate([x_orig, y_orig, [1.]])
    v = np.zeros_like(u)
    v[n:-1] = s_orig

    # for _ in range(5):#n + m):
    #     u, v = refine(
    #         z=u-v, matrix=matrix, b=b, c=c, zero=zero,
    #         nonneg=nonneg)#, max_iters=30)

    # # Apply HSDE scaling
    # x = u1 / u3
    # y = u2 / u3
    # s = v2 / u3

    return x_orig, y_orig, s_orig
