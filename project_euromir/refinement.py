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
"""Refinement function, simple Python implementation for now."""

import time

import numpy as np
import scipy as sp


def refine(z, matrix, b, c, zero, nonneg, max_iters=None):
    """All Python for now, will call all the rest.

    Equilibration is not done here, data must be already transformed.
    """

    # some shape checking
    n = len(c)
    m = len(b)
    assert matrix.shape == (m, n)
    assert zero + nonneg == m

    # temporary, build sparse Q
    Q = sp.sparse.bmat([
        [None, matrix.T, c.reshape(n, 1)],
        [-matrix, None, b.reshape(m, 1)],
        [-c.reshape(1, n), -b.reshape(1, m), None],
        ]).tocsc()

    # define convenience function
    def project(variable):
        result = np.copy(variable)
        result[n+zero:] = np.maximum(result[n+zero:], 0.)
        return result

    # compute u and v by projection
    u = project(z)
    v = u - z

    # very basic, for LPs just compute DR as sparse matrix
    mask = np.ones(len(z), dtype=float)
    mask[n+zero:] = z[n+zero:] > 0
    DR = (Q - sp.sparse.eye(Q.shape[0])) @ sp.sparse.diags(mask
        ) + sp.sparse.eye(Q.shape[0])

    # obtain initial (pre-LSQR) residual
    residual = Q @ u - v
    oldnormsq = np.linalg.norm(residual)**2
    print('residual norm sq before LSQR', oldnormsq)
    print(f'kappa = {u[-1]:.1e}, tau = {v[-1]:.1e}')

    # call LSQR
    start = time.time()
    result = sp.sparse.linalg.lsqr(
        DR, residual, atol=0., btol=0.,
        iter_lim=(Q.shape[0]*2) if max_iters is None else max_iters)
    print('LSQR result[1:-1]', result[1:-1])
    print('LSQR took', time.time() - start)
    dz = result[0]

    # recompute problem variables
    for i in range(20):
        z1 = z - dz * (0.5) ** i
        u = project(z1)
        v = u - z1
        newnormsq = np.linalg.norm(Q @ u - v)**2
        if newnormsq < oldnormsq:
            break
    else:
        print('REFINEMENT FAILED!')

    print("residual norm sq after LSQR", newnormsq)
    print(f'kappa = {u[-1]:.1e}, tau = {v[-1]:.1e}')
    return u, v
