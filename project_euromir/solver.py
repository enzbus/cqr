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

PYTHON_LBFGS = False

def solve(matrix, b, c, zero, nonneg):
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
    def loss_gradient(variable):
        residual[:] = system_matrix @ variable
        error[:] = np.minimum(variable[n+zero:], 0)
        loss = np.linalg.norm(residual)**2 + np.linalg.norm(error)**2
        gradient[:] = 2 * (system_matrix.T @ residual)
        gradient[n+zero:] += 2 * error
        return loss, gradient

    # initialize with all zeros and 1 only on the HSDE feasible flag
    x_0 = np.zeros(system_matrix.shape[1])
    x_0[n+m] = 1.

    # debug mode, plot history of losses
    if DEBUG:
        residual_sqnorms = []
        violation_sqnorms = []
        def _callback(variable):
            residual[:] = system_matrix @ variable
            error[:] = np.minimum(variable[n+zero:], 0)
            residual_sqnorms.append(np.linalg.norm(residual)**2)
            violation_sqnorms.append(np.linalg.norm(error)**2)

    # call LBFGS
    start = time.time()
    if PYTHON_LBFGS:
        lbfgs_result = sp.optimize.fmin_l_bfgs_b(
            loss_gradient,
            x0=x_0,
            m=10,
            maxfun=1e10,
            factr=0.,
            pgtol=1e-16, # e.g., simply use this for termination
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
            memory=10,
            max_iters=100000,
            c_1=1e-3, c_2=.9,
            ls_backtrack=.5,
            ls_forward=1.1,
            max_ls=100,
            use_active_set = False)
    print('LBFGS took', time.time() - start)

    # debug mode, plot history of losses
    if DEBUG:
        plt.plot(residual_sqnorms, label='residual square norms')
        plt.plot(violation_sqnorms, label='violation square norms')
        plt.yscale('log')
        plt.legend()
        plt.show()

    # extract result
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
