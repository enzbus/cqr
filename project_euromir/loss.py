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

HSDE_SCALING_PENALTY = False

###
# The loss function used in the main loop
###

def create_workspace_main(Q, n, zero, nonneg):
    """Create workspace used in main loop."""
    workspace = {}
    len_var = n + zero + nonneg + 1
    workspace['u_conic_error'] = np.empty(nonneg+1, dtype=float)
    workspace['v_conic_error'] = np.empty(len_var, dtype=float)
    workspace['gradient'] = np.empty(len_var, dtype=float)
    if HSDE_SCALING_PENALTY:
        workspace['scaler_loss'] = None
        workspace['scaler'] = Q[-1].todense().A1
        workspace['scaler'][-1] = 1.
    return workspace

def common_computation_main(u, Q, n, zero, nonneg, workspace):
    """Do common computation, in workspace, for input u."""
    workspace['u_conic_error'][:] = np.minimum(u[n+zero:], 0.)
    workspace['v_conic_error'][:] = Q @ u
    if HSDE_SCALING_PENALTY:
        workspace['v_m_1'] = workspace['v_conic_error'][-1]
    workspace['v_conic_error'][n+zero:] = np.minimum(
        workspace['v_conic_error'][n+zero:], 0.)

def loss(u, Q, n, zero, nonneg, workspace):
    """Compute loss from workspace."""
    result = (np.linalg.norm(workspace['u_conic_error'])**2
        + np.linalg.norm(workspace['v_conic_error'])**2)
    if HSDE_SCALING_PENALTY:
        workspace['scaler_loss'] = (workspace['scaler'] @ u - 1)**2
        result += workspace['scaler_loss']
    return result

def gradient(u, Q, n, zero, nonneg, workspace):
    """Compute gradient from workspace.

    We return the pre-allocated one, should be OK.
    """
    workspace['gradient'][:] = 2 * (Q.T @ workspace['v_conic_error'])
    workspace['gradient'][n+zero:] += 2 * workspace['u_conic_error']
    if HSDE_SCALING_PENALTY:
        workspace['gradient'][:] += 2 * workspace['scaler'] * workspace['scaler_loss']
    return workspace['gradient']

def hessian(u, Q, n, zero, nonneg, workspace):
    """Hessian LinearOperator."""

    def _matvec(myvar):

        tmp = Q @ myvar
        tmp[n+zero:] *= (workspace['v_conic_error'][n+zero:] < 0)
        tmp = Q.T @ tmp
        tmp[n+zero:] += (workspace['u_conic_error'] < 0) * myvar[n+zero:]
        if HSDE_SCALING_PENALTY:
            tmp[:] += workspace['scaler'] * (workspace['scaler'] @ myvar)
        return 2 * tmp

    return sp.sparse.linalg.LinearOperator(
        shape=(len(u), len(u)), matvec=_matvec)

def _densify(linear_operator):
    assert linear_operator.shape[0] == linear_operator.shape[1]
    result = np.empty(linear_operator.shape)
    identity = np.eye(result.shape[0])
    for i in range(len(identity)):
        result[:, i] = linear_operator.matvec(identity[:, i])
    return result

def _rdensify(linear_operator):
    assert linear_operator.shape[0] == linear_operator.shape[1]
    result = np.empty(linear_operator.shape)
    identity = np.eye(result.shape[0])
    for i in range(len(identity)):
        result[:, i] = linear_operator.rmatvec(identity[:, i])
    return result.T


###
# The residual and linear operator used in refinement
###

def create_workspace_refinement(n, zero, nonneg):
    """Create workspace used in refinement loop."""
    workspace = {}
    len_var = n + zero + nonneg + 1
    workspace['u_cone'] = np.empty(nonneg + 1, dtype=float)
    workspace['v_cone'] = np.empty(nonneg + 1, dtype=float)
    workspace['tmp'] = np.empty(nonneg + 1, dtype=float)
    workspace['residual'] = np.empty(len_var, dtype=float)
    return workspace

def common_computation_refinement(z, Q, n, zero, nonneg, workspace):
    """Do common computation, in workspace, for input z."""
    workspace['u_cone'][:] = np.maximum(z[n+zero:], 0.)
    workspace['v_cone'][:] = workspace['u_cone'] - z[n+zero:]

def residual(z, Q, n, zero, nonneg, workspace):

    # first piece of z is equal to first piece of u
    workspace['residual'][:] = Q[:, :n+zero] @ z[:n+zero]

    # second piece of u
    workspace['residual'][:] += Q[:, n+zero:] @ workspace['u_cone']

    # subtract nonzero conic part of v (first piece is zero)
    workspace['residual'][n+zero:] -= workspace['v_cone']

    # we can split the subtraction, saves memory; we may lose num accuracy
    # workspace['residual'][n+zero:] -= workspace['u_cone']
    # workspace['residual'][n+zero:] += z[n+zero:]

    return workspace['residual']

def _project(variable, n, zero):
    result = np.copy(variable)
    result[n+zero:] = np.maximum(result[n+zero:], 0.)
    return result

def _residual_basic(z, Q, n, zero, nonneg, **kwargs):
    # compute u and v by projection
    u = _project(z, n, zero)
    v = u - z
    return Q @ u - v

def _derivative_residual_basic(z, Q, n, zero, nonneg, **kwargs):
    # very basic, for LPs just compute DR as sparse matrix
    mask = np.ones(len(z), dtype=float)
    mask[n+zero:] = z[n+zero:] > 0
    return (Q - sp.sparse.eye(Q.shape[0])) @ sp.sparse.diags(mask
        ) + sp.sparse.eye(Q.shape[0])

def derivative_residual(z, Q, n, zero, nonneg, workspace):
    """Derivative of the residual operator."""

    def _matvec(myvar):
        result = np.empty_like(myvar)

        # apply derivative of Pi
        workspace['tmp'][:] = (workspace['u_cone'] > 0) * myvar[n+zero:]

        # apply Q
        result[:] = Q[:, n+zero:] @ workspace['tmp']

        # subtract internal identity
        result[n+zero:] -= workspace['tmp']

        # add external identity
        result[n+zero:] += myvar[n+zero:]

        # add zero cone part
        result[:] += Q[:, :n+zero] @ myvar[:n+zero]

        return result

    def _rmatvec(myvar):
        result = np.empty_like(myvar)

        # we can multiply by Q all together
        result[:] = myvar @ Q

        # nonzero conic part, subtract internal identity
        result[n+zero:] -= myvar[n+zero:]

        # multiply by DR
        result[n+zero:] *= (workspace['u_cone'] > 0)

        # add external identity
        result[n+zero:] += myvar[n+zero:]

        return result

    return sp.sparse.linalg.LinearOperator(
        shape=(len(z), len(z)), matvec=_matvec, rmatvec=_rmatvec)


if __name__ == '__main__':

    from scipy.optimize import check_grad

    np.random.seed(0)
    m = 20
    n = 10
    zero = 5
    nonneg = 15
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    c = np.random.randn(n)
    Q = sp.sparse.bmat([
        [None, A.T, c.reshape(n, 1)],
        [-A, None, b.reshape(m, 1)],
        [-c.reshape(1, n), -b.reshape(1, m), None],
        ]).tocsc()

    workspace = create_workspace_main(Q, n, zero, nonneg)
    u = np.random.randn(m+n+1)
    common_computation_main(u, Q, n, zero, nonneg, workspace)

    def my_loss(u):
        common_computation_main(u, Q, n, zero, nonneg, workspace)
        return loss(u, Q, n, zero, nonneg, workspace)

    def my_grad(u):
        common_computation_main(u, Q, n, zero, nonneg, workspace)
        return np.copy(gradient(u, Q, n, zero, nonneg, workspace))

    def my_hessian(u):
        common_computation_main(u, Q, n, zero, nonneg, workspace)
        return _densify(hessian(u, Q, n, zero, nonneg, workspace))

    print('\nCHECKING GRADIENT')
    for i in range(10):
        print(check_grad(my_loss, my_grad, np.random.randn(n+m+1)))

    print('\nCHECKING HESSIAN')
    for i in range(10):
        print(check_grad(my_grad, my_hessian, np.random.randn(n+m+1)))

    workspace_residual = create_workspace_refinement(n, zero, nonneg)

    def my_residual(z):
        common_computation_refinement(
            z, Q, n, zero, nonneg, workspace_residual)
        return np.copy(residual(z, Q, n, zero, nonneg, workspace_residual))

    def my_derivative_residual(z):
        common_computation_refinement(
            z, Q, n, zero, nonneg, workspace_residual)
        return _densify(derivative_residual(
            z, Q, n, zero, nonneg, workspace_residual))

    def my_derivative_residual_right(z):
        common_computation_refinement(
            z, Q, n, zero, nonneg, workspace_residual)
        return _rdensify(derivative_residual(
            z, Q, n, zero, nonneg, workspace_residual))

    print('\nCHECKING RESIDUAL')
    for i in range(10):
        z = np.random.randn(n+m+1)
        assert np.allclose(
            my_residual(z), _residual_basic(z, Q, n, zero, nonneg))
        assert np.allclose(
            my_derivative_residual(z),
            _derivative_residual_basic(z, Q, n, zero, nonneg).todense())
        print('left:\t', end='')
        print(check_grad(my_residual, my_derivative_residual, z))
        print('right:\t', end='')
        print(check_grad(my_residual, my_derivative_residual_right, z))
        assert np.allclose(my_derivative_residual_right(z), my_derivative_residual(z))
