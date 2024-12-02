# Copyright 2024 Enzo Busseti
#
# This file is part of Project Euromir.
#
# Project Euromir is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Project Euromir is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Project Euromir. If not, see <https://www.gnu.org/licenses/>.
"""Solver main function, will be unpacked and call all the rest."""

import time

import numpy as np
import scipy as sp

from project_euromir import equilibrate
from project_euromir.lbfgs import minimize_lbfgs
from project_euromir.refinement import refine

DEBUG = False
SCIPY_LBFGS = False
if DEBUG:
    import matplotlib.pyplot as plt

QR_PRESOLVE = False

def solve(
    matrix, b, c, zero, nonneg, lbfgs_memory=10, refinement_iters=50,
    refinement_lsqr_iters=100):
    "Main function."

    # some shape checking
    n = len(c)
    m = len(b)
    assert matrix.shape == (m, n)
    assert zero + nonneg == m

    if QR_PRESOLVE:
        q, r = np.linalg.qr(np.vstack([matrix.todense(), c.reshape((1, n))]))
        matrix_transf = q[:-1].A
        c_transf = q[-1].A1
        sigma = np.linalg.norm(b)
        b_transf = b/sigma

    else:
        # equilibration
        d, e, sigma, rho, matrix_transf, b_transf, c_transf = \
        equilibrate.hsde_ruiz_equilibration(
                matrix, b, c, dimensions={
                    'zero': zero, 'nonneg': nonneg, 'second_order': ()},
                max_iters=25)

    # define (loss, gradient) function; this is for LPs only

    # preallocate some variables
    y_error = np.empty(m-zero, dtype=float)
    s_error = np.empty(m, dtype=float)
    dual_residual = np.empty(n, dtype=float)
    s = np.empty(m, dtype=float)
    gradient = np.empty(n+m, dtype=float)

    # variable is xy
    def loss_gradient(xy):
        """Function for LBFGS loop, used in line search as well."""

        x = xy[:n]
        y = xy[n:]

        # zero cone dual variable is unconstrained
        y_error[:] = np.minimum(y[zero:], 0.)

        # this must be all zeros
        dual_residual[:] = matrix_transf.T @ y + c_transf

        # slacks
        s[:] = -matrix_transf @ x + b_transf

        # slacks for zero cone must be zero
        s_error[:zero] = s[:zero]
        s_error[zero:] = np.minimum(s[zero:], 0.)

        # duality gap
        gap = c_transf.T @ x + b_transf.T @ y

        # loss
        loss = np.linalg.norm(y_error)**2
        loss += np.linalg.norm(dual_residual)**2
        loss += np.linalg.norm(s_error)**2
        loss += gap**2

        # dual residual sqnorm
        gradient[n:] = 2 * (matrix_transf @ dual_residual)

        # s_error sqnorm
        gradient[:n] = -2 * (matrix_transf.T @ s_error)

        # y_error sqnorm
        gradient[n+zero:] += 2 * y_error

        # gap sq
        gradient[:n] += (2 * gap) * c_transf
        gradient[n:] += (2 * gap) * b_transf

        return loss, gradient

    def hessian(xy):
        """Hessian to use inside LBFGS loop."""

        x = xy[:n]
        y = xy[n:]

        # zero cone dual variable is unconstrained
        y_error[:] = np.minimum(y[zero:], 0.)

        # this must be all zeros
        dual_residual[:] = matrix_transf.T @ y + c_transf

        # slacks
        s[:] = -matrix_transf @ x + b_transf

        # slacks for zero cone must be zero
        s_error[:zero] = s[:zero]
        s_error[zero:] = np.minimum(s[zero:], 0.)

        def _matvec(dxdy):
            result = np.empty_like(dxdy)
            dx = dxdy[:n]
            dy = dxdy[n:]

            # dual residual sqnorm
            result[n:] = 2 * (matrix_transf @ (matrix_transf.T @ dy))

            # s_error sqnorm
            s_mask = np.ones(m, dtype=float)
            s_mask[zero:] = s_error[zero:] < 0.
            result[:n] = 2 * (matrix_transf.T @ (s_mask * (matrix_transf @ dx)))

            # y_error sqnorm
            y_mask = np.ones(m-zero, dtype=float)
            y_mask[:] = y_error < 0.
            result[n+zero:] += 2 * y_mask * dy[zero:]

            # gap
            constants = np.concatenate([c_transf, b_transf])
            result[:] += constants * (2 * (constants @ dxdy))

            return result

        return sp.sparse.linalg.LinearOperator(
            shape=(len(xy), len(xy)),
            matvec=_matvec
        )

    def callback(xy):
        loss, _ = loss_gradient(xy)
        # loss = loss_gradient(xy)
        print(loss)

    # initial variable, can pass initial guess
    x_0 = np.zeros(n+m)

    # call LBFGS
    start = time.time()

    if SCIPY_LBFGS:
        lbfgs_result = sp.optimize.fmin_l_bfgs_b(
            loss_gradient,
            x0=x_0,
            m=lbfgs_memory,
            # approx_grad=True,
            maxfun=1e10,
            factr=0.,
            pgtol=0.,
            callback=callback if DEBUG else None,
            maxiter=1e10)

        # print LBFGS stats
        function_value = lbfgs_result[1]
        print('LOSS AT THE END OF LBFGS', function_value)
        stats = lbfgs_result[2]
        stats.pop('grad')
        print('LBFGS stats', stats)
        result_variable = lbfgs_result[0]

    else:
        result_variable = minimize_lbfgs(
            loss_and_gradient_function=loss_gradient,
            initial_point=x_0,
            # callback=_callback if DEBUG else None,
            memory=lbfgs_memory,
            max_iters=int(1e10),
            #c_1=1e-3,
            #c_2=.1,
            # ls_backtrack=.5,
            # ls_forward=1.1,
            pgtol=0., #PGTOL,
            # hessian_approximator=hessian,
            #hessian_cg_iters=20,
            # use_active_set=ACTIVE_SET,
            max_ls=100)

    print('LBFGS took', time.time() - start)

    # refinement, still based on hsde
    Q = sp.sparse.bmat([
        [None, matrix_transf.T, c_transf.reshape(n, 1)],
        [-matrix_transf, None, b_transf.reshape(m, 1)],
        [-c_transf.reshape(1, n), -b_transf.reshape(1, m), None],
        ]).tocsc()
    u = np.empty(n+m+1)

    # create hsde variables
    u[:-1] = result_variable
    u[-1] = 1.
    v = Q @ u

    # refine
    for i in range(refinement_iters):
        u, v = refine(
            z=u-v, matrix=matrix_transf, b=b_transf, c=c_transf, zero=zero,
            nonneg=nonneg, max_iters=refinement_lsqr_iters)

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
        # invert Ruiz scaling, copied from other repo
        x_orig =  e * x / sigma
        y_orig = d * y / rho
        s_orig = (s/d) / sigma

    return x_orig, y_orig, s_orig
