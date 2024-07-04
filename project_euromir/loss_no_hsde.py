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
"""Define loss function(s), gradient(s), ...."""

import numpy as np
import scipy as sp

###
# The loss function used in the main loop
###

def create_workspace(m, n, zero):
    workspace = {}

    # preallocate some variables
    workspace['y_error'] = np.empty(m-zero, dtype=float)
    workspace['s_error'] = np.empty(m, dtype=float)
    workspace['dual_residual'] = np.empty(n, dtype=float)
    workspace['s'] = np.empty(m, dtype=float)
    workspace['gradient'] = np.empty(n+m, dtype=float)

    return workspace

# variable is xy
def loss_gradient(xy, m, n, zero, matrix, b, c, workspace):
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

    # dual residual sqnorm
    workspace['gradient'][n:] = 2 * (matrix @ workspace['dual_residual'])

    # s_error sqnorm
    workspace['gradient'][:n] = -2 * (matrix.T @ workspace['s_error'])

    # y_error sqnorm
    workspace['gradient'][n+zero:] += 2 * workspace['y_error']

    # gap sq
    workspace['gradient'][:n] += (2 * gap) * c
    workspace['gradient'][n:] += (2 * gap) * b

    return loss, workspace['gradient']

def hessian(xy, m, n, zero, matrix, b, c, workspace, regularizer = 0.):
    """Hessian to use inside LBFGS loop."""

    locals().update(workspace)

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
        result[n:] = 2 * (matrix @ (matrix.T @ dy))

        # s_error sqnorm
        s_mask = np.ones(m, dtype=float)
        s_mask[zero:] = workspace['s_error'][zero:] < 0.
        result[:n] = 2 * (matrix.T @ (s_mask * (matrix @ dx)))

        # y_error sqnorm
        y_mask = np.ones(m-zero, dtype=float)
        y_mask[:] = workspace['y_error'] < 0.
        result[n+zero:] += 2 * y_mask * dy[zero:]

        # gap
        constants = np.concatenate([c, b])
        result[:] += constants * (2 * (constants @ dxdy))

        return result + regularizer * dxdy

    return sp.sparse.linalg.LinearOperator(
        shape=(len(xy), len(xy)),
        matvec=_matvec
    )

def _densify(linear_operator):
    assert linear_operator.shape[0] == linear_operator.shape[1]
    result = np.empty(linear_operator.shape)
    identity = np.eye(result.shape[0])
    for i in range(len(identity)):
        result[:, i] = linear_operator.matvec(identity[:, i])
    return result


if __name__ == '__main__':

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
        return _densify(hessian(xy, m, n, zero, matrix, b, c, wks))

    print('\nCHECKING GRADIENT')
    for i in range(10):
        print(check_grad(my_loss, my_grad, np.random.randn(n+m)))

    print('\nCHECKING HESSIAN')
    for i in range(10):
        print(check_grad(my_grad, my_hessian, np.random.randn(n+m)))
