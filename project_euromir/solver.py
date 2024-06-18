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

from project_euromir import equilibrate
from project_euromir.lbfgs import minimize_lbfgs
from project_euromir.refinement import refine

DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt

USE_MY_LBFGS = True
ACTIVE_SET = False # this doesn't work yet, not sure if worth trying to fix it
IMPLICIT_FORMULATION = False # this does help!!! some minor issues on hessian

if ACTIVE_SET:
    assert USE_MY_LBFGS

PGTOL = 0. # I tried this as a stopping condition for lbfgs, but it can break
# (meaning that active set is still not robust and when switching to lsqr
# it breaks); you can try e.g. 1e-12

def solve(matrix, b, c, zero, nonneg, lbfgs_memory=10):
    "Main function."

    # some shape checking
    n = len(c)
    m = len(b)
    assert matrix.shape == (m, n)
    assert zero + nonneg == m

    # equilibration
    d, e, sigma, rho, matrix_transf, b_transf, c_transf = \
    equilibrate.hsde_ruiz_equilibration(
            matrix, b, c, dimensions={
                'zero': zero, 'nonneg': nonneg, 'second_order': ()},
            max_iters=25)

    # temporary, build sparse Q
    Q = sp.sparse.bmat([
        [None, matrix_transf.T, c_transf.reshape(n, 1)],
        [-matrix_transf, None, b_transf.reshape(m, 1)],
        [-c_transf.reshape(1, n), -b_transf.reshape(1, m), None],
        ]).tocsc()

    # temporary, [Q, -I]
    QI = sp.sparse.hstack([Q, -sp.sparse.eye(n+m+1, format='csc')])

    # temporary, remove v in zero cone
    _as = np.concatenate(
        [np.ones(n+m+1, dtype=bool),
        np.zeros(n + zero, dtype=bool),
        np.ones(m+1 - zero, dtype=bool)])

    # so we define the matrix of the LBFGS loop
    system_matrix = QI[:, _as]

    # pre-allocate vars used below
    residual = np.empty(system_matrix.shape[0], dtype=float)
    error = np.empty(system_matrix.shape[1]-n-zero, dtype=float)
    gradient = np.empty(system_matrix.shape[1], dtype=float)

    # temporary, just define loss-gradient function for LPs

    if ACTIVE_SET:
        raise Exception('Need to remove this option.')
        # test using active set instead and internal projection
        # when extending to other cones we'll have to figure out
        def loss_gradient(variable):
            variable[n+zero:] = np.maximum(variable[n+zero:], 0)
            residual[:] = system_matrix @ variable
            # error[:] = np.minimum(variable[n+zero:], 0)
            loss = np.linalg.norm(residual)**2 #+ np.linalg.norm(error)**2
            gradient[:] = 2 * (system_matrix.T @ residual)
            active_set = np.ones_like(variable, dtype=bool)
            active_set[n+zero:] = (variable[n+zero:] > 0) | (gradient[n+zero:] < 0)
            # gradient[n+zero:] += 2 * error
            return loss, gradient, active_set

    else:
        if IMPLICIT_FORMULATION:
            # variable is only u
            def loss_gradient(u):
                resu = np.minimum(u[n+zero:], 0.)
                resv1 = np.minimum(Q[n+zero:] @ u, 0.)
                resv2 = Q[:n+zero] @ u
                loss = np.linalg.norm(resu)**2
                loss += np.linalg.norm(resv1)**2
                loss += np.linalg.norm(resv2)**2

                grad = np.zeros_like(u)
                grad[n+zero:] += 2 * resu
                grad += 2 * Q[n+zero:].T @ resv1
                grad += 2 * Q[:n+zero].T @ resv2

                return loss, grad

            def hessian(u): # TODO: this is not correct yet, need to check_grad it (it's close)
                def _matvec(myvar):
                    result = np.zeros_like(u)

                    #resu
                    result[n+zero:][u[n+zero:] < 0] += myvar[n+zero:][u[n+zero:] < 0]

                    #resv1
                    resv1_nonproj = Q[n+zero:] @ u
                    tmp = Q[n+zero:] @ myvar
                    tmp[resv1_nonproj > 0] = 0.
                    result += Q[n+zero:].T @ tmp

                    #resv2
                    result += Q[:n+zero].T @ (Q[:n+zero] @ myvar)

                    return 2 * result
                return sp.sparse.linalg.LinearOperator(
                    shape=(len(u), len(u)),
                    matvec=_matvec
                )

        else:
            def loss_gradient(variable):
                residual[:] = system_matrix @ variable
                error[:] = np.minimum(variable[n+zero:], 0)
                loss = np.linalg.norm(residual)**2 + np.linalg.norm(error)**2
                gradient[:] = 2 * (system_matrix.T @ residual)
                gradient[n+zero:] += 2 * error
                return loss, gradient

            def hessian(variable):
                def _matvec(myvar):
                    result = system_matrix.T @ (system_matrix @ myvar)
                    result[n+zero:][variable[n+zero:] < 0] += myvar[n+zero:][variable[n+zero:] < 0]
                    return 2 * result
                return sp.sparse.linalg.LinearOperator(
                    shape=(len(variable), len(variable)),
                    matvec=_matvec
                )

    if IMPLICIT_FORMULATION:
        x_0 = np.zeros(n+m+1)
        x_0[-1] = 1.
    else:
        # initialize with all zeros and 1 only on the HSDE feasible flag
        x_0 = np.zeros(system_matrix.shape[1])
        x_0[n+m] = 1.

    # debug mode, plot history of losses
    if DEBUG:
        residual_sqnorms = []
        violation_sqnorms = []
        def _callback(variable):
            assert not IMPLICIT_FORMULATION
            residual[:] = system_matrix @ variable
            error[:] = np.minimum(variable[n+zero:], 0)
            residual_sqnorms.append(np.linalg.norm(residual)**2)
            violation_sqnorms.append(np.linalg.norm(error)**2)

    # call LBFGS
    start = time.time()
    if not USE_MY_LBFGS:
        lbfgs_result = sp.optimize.fmin_l_bfgs_b(
            loss_gradient,
            x0=x_0,
            m=lbfgs_memory,
            maxfun=1e10,
            factr=0.,
            pgtol=PGTOL,
            callback=_callback if DEBUG else None,
            maxiter=1e10)
        # print LBFGS stats
        stats = lbfgs_result[2]
        stats.pop('grad')
        print('LBFGS stats', stats)
        result_variable = lbfgs_result[0]
    else:
        result_variable = minimize_lbfgs(
            loss_and_gradient_function=loss_gradient,
            initial_point=x_0,
            callback=_callback if DEBUG else None,
            memory=lbfgs_memory,
            max_iters=int(1e10),
            # c_1=1e-3, c_2=.9,
            # ls_backtrack=.5,
            # ls_forward=1.1,
            pgtol=PGTOL,
            hessian_approximator=hessian,
            use_active_set=ACTIVE_SET,
            max_ls=100)
    print('LBFGS took', time.time() - start)

    # debug mode, plot history of losses
    if DEBUG:
        plt.plot(residual_sqnorms, label='residual square norms')
        plt.plot(violation_sqnorms, label='violation square norms')
        plt.yscale('log')
        plt.legend()
        plt.show()

    # extract result
    if IMPLICIT_FORMULATION:
        u = result_variable
        assert u[-1] > 0, 'Certificates not yet implemented'
        u /= u[-1]
        v = Q @ u
    else:
        u = result_variable[:n+m+1]
        v = np.zeros(n+m+1)
        v[n+zero:] = result_variable[n+m+1:]

    u, v = refine(
        z=u-v, matrix=matrix_transf, b=b_transf, c=c_transf, zero=zero,
        nonneg=nonneg)

    # Transform back into problem format
    u1, u2, u3 = u[:n], u[n:n+m], u[-1]
    v2, v3 = v[n:n+m], v[-1]

    if v3 > u3:
        raise NotImplementedError('Certificates not yet implemented.')

    # Apply HSDE scaling
    x = u1 / u3
    y = u2 / u3
    s = v2 / u3

    # invert Ruiz scaling, copied from other repo
    x_orig =  e * x / sigma
    y_orig = d * y / rho
    s_orig = (s/d) / sigma

    return x_orig, y_orig, s_orig
