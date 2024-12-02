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
from project_euromir.loss_no_hsde import (_densify, create_workspace, hessian,
                                          loss_gradient)
from project_euromir.newton_cg import _epsilon, _minimize_newtoncg
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

    # temporary, build sparse Q
    Q = sp.sparse.bmat([
        [None, matrix_transf.T, c_transf.reshape(n, 1)],
        [-matrix_transf, None, b_transf.reshape(m, 1)],
        [-c_transf.reshape(1, n), -b_transf.reshape(1, m), None],
        ]).tocsc()

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
            xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf, c=c_transf, workspace=workspace)

    x_0 = np.zeros(n+m)

    start = time.time()
    xy = np.array(x_0)
    import matplotlib.pyplot as plt
    for i in range(1000):
        print('iter', i)
        g = separated_grad(xy)
        H = _densify(separated_hessian(xy))
        p = np.linalg.pinv(H) @ -g
        steps = np.linspace(0, 1, 101)
        losses = [separated_loss(xy + step * p) for step in steps]
        # plt.plot(steps, losses)
        # plt.show()
        best_step = steps[np.argmin(losses)]
        print('loss', np.min(losses))
        print('step', best_step)
        if best_step == 0.:
            xy -= g/100
        else:
            xy += best_step * p
        if np.min(losses) < 2 * np.finfo(float).eps**2:
            break
        # breakpoint()

    print(f'NEWTON-CG took {time.time() - start:.3f} seconds')

    u = np.empty(n+m+1, dtype=float)
    u[:-1] = xy
    u[-1] = 1.

    # u /= u[-1]
    v = Q @ u

    u, v = refine(
        z=u-v, matrix=matrix_transf, b=b_transf, c=c_transf, zero=zero,
        nonneg=nonneg)

    # u, v = refine(
    #     z=u-v, matrix=matrix_transf, b=b_transf, c=c_transf, zero=zero,
    #     nonneg=nonneg)

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
