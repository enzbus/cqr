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
"""Define residual and Dresidual for use by refinement loop."""

import numpy as np
import scipy as sp

from .loss_no_hsde import _densify_also_nonsquare


def residual(xz, m, n, zero, nonneg, matrix, b, c, soc=()):
    """Residual function for refinement."""

    x = xz[:n]
    z = xz[n:]

    # projection
    y = np.empty_like(z)
    y[:zero] = z[:zero]
    y[zero:zero+nonneg] = np.maximum(z[zero:zero+nonneg], 0.)
    s = y - z

    # print(y)
    # print(s)

    # primal residual
    primal_residual = matrix @ x - b + s

    # dual residual
    dual_residual = c + matrix.T @ y

    # duality gap
    gap = c.T @ x + b.T @ y

    # build the full residual by concatenating residuals
    res = np.zeros(n + m + 1, dtype=float)
    res[:m] = primal_residual
    res[m:m+n] = dual_residual
    res[-1] = gap

    return res

def Dresidual_densefull(xz, m, n, zero, nonneg, matrix, b, c, soc=()):
    """Dense Jacobian for testing."""

    jacobian = np.zeros((n+m+1, n+m), dtype=float)

    assert len(soc) == 0

    # Pi derivatives
    y_mask = np.ones(m, dtype=float)
    y_mask[zero:zero+nonneg] = xz[n+zero:n+zero+nonneg] >= 0
    s_mask = y_mask - 1.
    # print(y_mask)
    # print(s_mask)

    # pri res
    jacobian[:m, :n] = matrix
    jacobian[:m, n:] = np.diag(s_mask)

    # dua res
    jacobian[m:m+n, n:] = matrix.T @ np.diag(y_mask)

    # gap
    jacobian[-1, :n] = c
    jacobian[-1, n:] = y_mask * b

    return jacobian


def Dresidual(xy, m, n, zero, nonneg, matrix, b, c, soc=()):
    """Linear operator to matrix multiply the refinement residual derivative."""

    x = xy[:n]
    y = xy[n:]

    # zero cone dual variable is unconstrained
    y_mask = (y[zero:] <= 0.) * 1.

    # slacks
    s = -matrix @ x + b

    # slacks for zero cone must be zero
    s_mask = np.ones_like(s)
    s_mask[zero:] = s[zero:] <= 0.

    # concatenation of primal and dual costs
    pridua = np.concatenate([c, b])

    def _matvec(dxy):

        # decompose direction
        dx = dxy[:n]
        dy = dxy[n:]

        # compose result
        dr = np.empty(n + 2 * m - zero + 1, dtype=float)
        dr[:m-zero] = y_mask * dy[zero:]
        dr[m-zero:m+n-zero] = matrix.T @ dy
        dr[-1-m:-1] = s_mask * (-(matrix @ dx))
        dr[-1] = pridua @ dxy

        return dr

    def _rmatvec(dr):

        # decompose direction
        dy_err = dr[:m-zero]
        ddua_res = dr[m-zero:m+n-zero]
        ds_err = dr[-1-m:-1]
        dgap = dr[-1]

        # compose result
        dxy = np.zeros(n + m, dtype=float)
        dxy[-(m-zero):] += y_mask * dy_err
        dxy[-m:] += matrix @ ddua_res
        dxy[:n] -= matrix.T @ (s_mask * ds_err)
        dxy += dgap * pridua

        return dxy

    return sp.sparse.linalg.LinearOperator(
        shape=(n + 2 * m - zero + 1, n+m),
        matvec = _matvec,
        rmatvec = _rmatvec)

if __name__ == '__main__': # pragma: no cover

    from scipy.optimize import check_grad

    # create consts
    np.random.seed(0)
    m = 20
    n = 10
    zero = 5
    nonneg = m-zero
    matrix = np.random.randn(m, n)
    b = np.random.randn(m)
    c = np.random.randn(n)

    def my_residual(xz):
        return residual(xz, m, n, zero, nonneg, matrix, b, c)

    def my_Dresidual(xz):
        return Dresidual(xz, m, n, zero, nonneg, matrix, b, c)

    def my_Dresidual_densefull(xz):
        return Dresidual_densefull(xz, m, n, zero, nonneg, matrix, b, c)

    def my_Dresidual_dense(xz):
        return _densify_also_nonsquare(my_Dresidual(xz))

    print('\nCHECKING D_RESIDUAL DENSE')
    for i in range(10):
        print(check_grad(
            my_residual, my_Dresidual_densefull, np.random.randn(n+m)))

    # print('\nCHECKING D_RESIDUAL')
    # for i in range(10):
    #     print(check_grad(my_residual, my_Dresidual_dense, np.random.randn(n+m)))

    # print('\nCHECKING DR and DR^T CONSISTENT')
    # for i in range(10):
    #     xy = np.random.randn(n+m)
    #     DR = _densify_also_nonsquare(my_Dresidual(xy))
    #     DRT = _densify_also_nonsquare(my_Dresidual(xy).T)
    #     assert np.allclose(DR.T, DRT)
    # print('\tOK!')
