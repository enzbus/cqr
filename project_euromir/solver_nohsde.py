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

        # return loss

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

    def callback(xy):
        loss, _ = loss_gradient(xy)
        # loss = loss_gradient(xy)
        print(loss)

    # initial variable, can pass initial guess
    x_0 = np.zeros(n+m)

    # call LBFGS
    start = time.time()
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
    for i in range(5):
        u, v = refine(
            z=u-v, matrix=matrix_transf, b=b_transf, c=c_transf, zero=zero,
            nonneg=nonneg, max_iters=100)

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
