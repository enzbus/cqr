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
"""Define loss function(s), gradient(s), ...."""

import numpy as np
import scipy as sp

###
# The loss function used in the main loop
###

def create_workspace(m, n, zero, soc=()):
    workspace = {}

    # preallocate some variables
    workspace['y_error'] = np.empty(m-zero, dtype=float)
    workspace['s_error'] = np.empty(m, dtype=float)
    workspace['dual_residual'] = np.empty(n, dtype=float)
    workspace['s'] = np.empty(m, dtype=float)
    workspace['gradient'] = np.empty(n+m, dtype=float)

    return workspace

# variable is xy
def loss_gradient(xy, m, n, zero, matrix, b, c, workspace, soc=()):
    """Function for LBFGS loop, used in line search as well."""

    x = xy[:n]
    y = xy[n:]

    # zero cone dual variable is unconstrained
    workspace['y_error'][:] = np.minimum(y[zero:], 0.)

    # this must be all zeros
    workspace['dual_residual'][:] = matrix.T @ y + c

    # slacks
    workspace['s'][:] = -matrix @ x + b

    # slacks for zero cone must be zero
    workspace['s_error'][:zero] = workspace['s'][:zero]
    workspace['s_error'][zero:] = np.minimum(workspace['s'][zero:], 0.)

    # duality gap
    gap = c.T @ x + b.T @ y

    # loss
    loss = np.linalg.norm(workspace['y_error'])**2
    loss += np.linalg.norm(workspace['dual_residual'])**2
    loss += np.linalg.norm(workspace['s_error'])**2
    loss += gap**2

    loss /= 2.

    # dual residual sqnorm
    workspace['gradient'][n:] = (matrix @ workspace['dual_residual'])

    # s_error sqnorm
    workspace['gradient'][:n] = -(matrix.T @ workspace['s_error'])

    # y_error sqnorm
    workspace['gradient'][n+zero:] += workspace['y_error']

    # gap sq
    workspace['gradient'][:n] += gap * c
    workspace['gradient'][n:] += gap * b

    return loss, workspace['gradient']

def hessian(xy, m, n, zero, matrix, b, c, workspace, soc=(), regularizer = 0.):
    """Hessian to use inside LBFGS loop."""

    x = xy[:n]
    y = xy[n:]

    # zero cone dual variable is unconstrained
    workspace['y_error'][:] = np.minimum(y[zero:], 0.)

    # this must be all zeros
    workspace['dual_residual'][:] = matrix.T @ y + c

    # slacks
    workspace['s'][:] = -matrix @ x + b

    # slacks for zero cone must be zero
    workspace['s_error'][:zero] = workspace['s'][:zero]
    workspace['s_error'][zero:] = np.minimum(workspace['s'][zero:], 0.)

    def _matvec(dxdy):
        result = np.empty_like(dxdy)
        dx = dxdy[:n]
        dy = dxdy[n:]

        # dual residual sqnorm
        result[n:] = (matrix @ (matrix.T @ dy))

        # s_error sqnorm
        s_mask = np.ones(m, dtype=float)
        s_mask[zero:] = workspace['s_error'][zero:] < 0.
        result[:n] = (matrix.T @ (s_mask * (matrix @ dx)))

        # y_error sqnorm
        y_mask = np.ones(m-zero, dtype=float)
        y_mask[:] = workspace['y_error'] < 0.
        result[n+zero:] += y_mask * dy[zero:]

        # gap
        constants = np.concatenate([c, b])
        result[:] += constants * (constants @ dxdy)

        return result + regularizer * dxdy

    return sp.sparse.linalg.LinearOperator(
        shape=(len(xy), len(xy)),
        matvec=_matvec
    )

def residual(xy, m, n, zero, matrix, b, c, soc=()):
    """Residual function to use L-M approach instead."""

    x = xy[:n]
    y = xy[n:]

    # zero cone dual variable is unconstrained
    y_error = np.minimum(y[zero:], 0.)

    # this must be all zeros
    dual_residual = matrix.T @ y + c

    # slacks
    s = -matrix @ x + b

    # slacks for zero cone must be zero
    s_error = np.copy(s)
    s_error[zero:] = np.minimum(s[zero:], 0.)

    # duality gap
    gap = c.T @ x + b.T @ y

    # build the full residual by concatenating residuals
    res = np.empty(n + 2 * m - zero + 1, dtype=float)
    res[:m-zero] = y_error
    res[m-zero:m+n-zero] = dual_residual
    res[-1-m:-1] = s_error
    res[-1] = gap

    return res

def Dresidual(xy, m, n, zero, matrix, b, c, soc=()):
    """Linear operator to matrix multiply the residual derivative."""

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

def _densify_also_nonsquare(linear_operator):
    result = np.empty(linear_operator.shape)
    for j in range(linear_operator.shape[1]):
        e_j = np.zeros(linear_operator.shape[1])
        e_j[j] = 1.
        result[:, j] = linear_operator.matvec(e_j)
    return result


if __name__ == '__main__': # pragma: no cover

    from scipy.optimize import check_grad

    # create consts
    np.random.seed(0)
    m = 20
    n = 10
    zero = 5
    nonneg = 15
    matrix = np.random.randn(m, n)
    b = np.random.randn(m)
    c = np.random.randn(n)

    wks = create_workspace(m, n, zero)

    def my_loss(xy):
        return loss_gradient(xy, m, n, zero, matrix, b, c, wks)[0]

    def my_grad(xy):
        return np.copy(loss_gradient(xy, m, n, zero, matrix, b, c, wks)[1])

    def my_hessian(xy):
        return _densify_also_nonsquare(
            hessian(xy, m, n, zero, matrix, b, c, wks))

    def my_residual(xy):
        return residual(xy, m, n, zero, matrix, b, c)

    def my_Dresidual(xy):
        return Dresidual(xy, m, n, zero, matrix, b, c)

    def my_hessian_from_dresidual(xy):
        DR = Dresidual(xy, m, n, zero, matrix, b, c)
        return sp.sparse.linalg.LinearOperator(
            (n+m, n+m),
            matvec = lambda dxy: DR.T @ (DR @ (dxy)))

    print('\nCHECKING GRADIENT')
    for i in range(10):
        print(check_grad(my_loss, my_grad, np.random.randn(n+m)))

    print('\nCHECKING HESSIAN')
    for i in range(10):
        print(check_grad(my_grad, my_hessian, np.random.randn(n+m)))

    print('\nCHECKING LOSS AND RESIDUAL CONSISTENT')
    for i in range(10):
        xy = np.random.randn(n+m)
        assert np.isclose(my_loss(xy), (np.linalg.norm(my_residual(xy))**2)/2.)
    print('\tOK!')

    print('\nCHECKING DR and DR^T CONSISTENT')
    for i in range(10):
        xy = np.random.randn(n+m)
        DR = _densify_also_nonsquare(my_Dresidual(xy))
        DRT = _densify_also_nonsquare(my_Dresidual(xy).T)
        assert np.allclose(DR.T, DRT)
    print('\tOK!')

    print('\nCHECKING (D)RESIDUAL AND GRADIENT CONSISTENT')
    for i in range(10):
        xy = np.random.randn(n+m)
        grad = my_grad(xy)
        newgrad = (my_Dresidual(xy).T @ my_residual(xy))
        assert np.allclose(grad, newgrad)
    print('\tOK!')

    print('\nCHECKING DRESIDUAL AND HESSIAN CONSISTENT')
    for i in range(10):
        xy = np.random.randn(n+m)
        hess = my_hessian(xy)
        hess_rebuilt = _densify_also_nonsquare(my_hessian_from_dresidual(xy))
        assert np.allclose(hess, hess_rebuilt)
    print('\tOK!')
