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
"""Application of the Minamide lemma to simplify solving the main system.

The Minamide lemma is a generalization of the matrix inversion lemma when
pseudo-solving a semi-definite symmetric system with a low-rank update.

Our rank-1 component are the program constants, which enter in the gap part of
the loss function. The rest of the system is semi-definite and separates
between the primal and dual sub-programs; It has also better conditioning than
the full system.

N. Minamide, 1985: https://epubs.siam.org/doi/abs/10.1137/0606038
"""

import numpy as np
import scipy as sp


def hessian_x_nogap(x, m, n, zero, matrix, b, regularizer = 0.):
    """Hessian of only the x part without the gap, with multiplication by 2."""

    # slacks; for zero cone must be zero
    s_error = -matrix @ x + b
    s_error[zero:] = np.minimum(s_error[zero:], 0.)

    # s_error sqnorm
    s_mask = np.ones(m, dtype=float)
    s_mask[zero:] = s_error[zero:] < 0.

    def _matvec(dx):
        result = np.empty_like(dx)
        result[:] = 2 * (matrix.T @ (s_mask * (matrix @ dx)))
        return result + regularizer * dx

    return sp.sparse.linalg.LinearOperator(
        shape=(len(x), len(x)),
        matvec=_matvec
    )

def hessian_y_nogap(y, m, n, zero, matrix, regularizer = 0.):
    """Hessian of only the y part without the gap, with multiplication by 2."""

    # zero cone dual variable is unconstrained
    y_error = np.minimum(y[zero:], 0.)
    y_mask = np.ones(m-zero, dtype=float)
    y_mask[:] = y_error < 0.

    def _matvec(dy):
        result = np.empty_like(dy)

        # dual residual sqnorm
        result[:] = 2 * (matrix @ (matrix.T @ dy))
        result[zero:] += 2 * y_mask * dy[zero:]

        return result + regularizer * dy

    return sp.sparse.linalg.LinearOperator(
        shape=(len(y), len(y)),
        matvec=_matvec
    )


from .direction_calculator import DirectionCalculator, _densify


class MinamideTest(DirectionCalculator):

    def __init__(self, b, c, h_x_nogap, h_y_nogap):
        self._constants_sqrt2 = np.concatenate([c, b]) * np.sqrt(2.)
        self._hessian_x_nogap = h_x_nogap
        self._hessian_y_nogap = h_y_nogap
        self._n = len(c)
        self._m = len(b)

    def get_direction(self, current_point, current_gradient):

        rhs = -current_gradient
        FINAL_REGU = 0. # 1e-12
        # with many tests I found instances where it
        # gets stuck on directions of little descent, not sure how much
        # regularization is needed

        # make all dense for test
        hx = _densify(self._hessian_x_nogap(current_point[:self._n]))#  + np.eye(self._n) * REGU
        hy = _densify(self._hessian_y_nogap(current_point[self._n:]))#  + np.eye(self._m) * REGU

        # minamide notation
        S = sp.sparse.bmat([
            [hx, None],
            [None, hy],
            ]).tocsc()
        phi = self._constants_sqrt2

        def _splus_matvec(array):
            return np.concatenate([
                np.linalg.lstsq(hx, array[:self._n], rcond=None)[0],
                np.linalg.lstsq(hy, array[self._n:], rcond=None)[0]
            ])

        # def _splus_matvec(array):
        #     return np.concatenate([
        #         sp.sparse.linalg.cg(hx, array[:self._n], rtol=1e-5)[0],
        #         sp.sparse.linalg.cg(hy, array[self._n:], rtol=1e-5)[0]
        #     ])

        # def _splus_matvec(array):
        #     return np.concatenate([
        #         sp.sparse.linalg.minres(hx, array[:self._n], rtol=min(0.5, np.linalg.norm(array[:self._n])**0.5))[0],
        #         sp.sparse.linalg.minres(hy, array[self._n:], rtol=min(0.5, np.linalg.norm(array[self._n:])**0.5))[0]
        #     ])

        Splus = sp.sparse.linalg.LinearOperator(
            shape = (self._n + self._m, self._n + self._m),
            matvec = _splus_matvec
        )

        Splusphi = Splus @ phi
        Tphi = phi - S @ Splusphi

        # breakpoint()
        # pinv_hx = np.linalg.pinv(hx)
        # pinv_hy = np.linalg.pinv(hy)
        # Splus = np.bmat([
        #     [pinv_hx, np.zeros((self._n, self._m))],
        #     [np.zeros((self._m, self._n)), pinv_hy],
        #     ]).A

        # Splus = np.linalg.pinv(S)

        # build T matrix (projector on null space of S)
        #T = np.eye(self._n+self._m) - S @ Splus

        # get part of phi in the nullspace
        #Tphi = T @ phi

        # simple case, ~Woodbury
        if np.allclose(Tphi, 0.):
            print('Case 2: phi orthogonal to the null space')

            # example 2.4 of paper, case 2.: Woodbury formula with pinv's
            # Splusphi = Splus @ phi
            #pinv_rebuilt = Splus - np.outer(Splusphi, Splusphi) / (
            #    1 + phi @ Splusphi)
            result = Splus @ rhs
            result -= Splusphi * ((Splusphi @ rhs) /  (1 + phi @ Splusphi))
            return result + FINAL_REGU * rhs

        # interesting case
        else:
            print('Case 1: phi not orthogonal to the null space')

            # example 2.4 of paper, case 1.
            #right = np.eye(self._n+self._m) - np.outer(phi, Tphi) / (phi @ Tphi)
            #left = right.T
            # extra = np.outer(Tphi, Tphi) / ((phi @ Tphi)**2)

            # multiply by right
            result = rhs - phi * ((Tphi @ rhs) / (phi @ Tphi))

            # multiply by Splus
            result = Splus @ result

            # multiply by left
            result = result - Tphi * ((phi @ result) / (phi @ Tphi))

            # extra term
            extra = Tphi * ((Tphi @ rhs) / ((phi @ Tphi)**2))

            #result = left @ (Splus @ (right @ rhs))
            return result + extra + FINAL_REGU * rhs

            # pinv_rebuilt = left @ Splus @ right + extra

        #import matplotlib.pyplot as plt
        #breakpoint()

        #return pinv_rebuilt @ rhs # - FINAL_REGU * current_gradient
        # pinv_hx = np.linalg.pinv(hx)
        # pinv_hy = np.linalg.pinv(hy)
        # pinv_base = np.bmat([
        #     [pinv_hx, np.zeros((self._n, self._m))],
        #     [np.zeros((self._m, self._n)), pinv_hy],
        #     ]).A

        # # obtain pinv of the reduced system
        # Splus = np.linalg.pinv(S)

        # # build T matrix (projector on null space of S)
        # T = np.eye(n+m) - Splus @ S

        # # do full psolve
        # _hessian = np.bmat([
        #     [hx, np.zeros((self._n, self._m))],
        #     [np.zeros((self._m, self._n)), hy],
        #     ]).A

        # _hessian += np.outer(self._constants_sqrt2, self._constants_sqrt2)

        # return np.linalg.lstsq(_hessian, -current_gradient, rcond=None)[0]


if __name__ == '__main__': # pragma: no cover

    from .loss_no_hsde import (_densify_also_nonsquare, create_workspace,
                               hessian)

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

    def my_hessian(xy):
        return _densify_also_nonsquare(
            hessian(xy, m, n, zero, matrix, b, c, wks))

    def my_hessian_x_nogap(x):
        return _densify_also_nonsquare(
            hessian_x_nogap(x, m, n, zero, matrix, b))

    def my_hessian_y_nogap(y):
        return _densify_also_nonsquare(
            hessian_y_nogap(y, m, n, zero, matrix))

    print('\nCHECKING DECOMPOSED HESSIAN CONSISTENT')
    for i in range(10):
        xy = np.random.randn(n+m)
        hess = my_hessian(xy)
        hess_x_ng = my_hessian_x_nogap(xy[:n])
        hess_y_ng = my_hessian_y_nogap(xy[n:])
        hess_rebuilt = np.bmat([
            [hess_x_ng, np.zeros((n, m))],
            [np.zeros((m, n)), hess_y_ng]])
        gap_constants = np.concatenate([c, b])
        hess_rebuilt += np.outer(gap_constants, gap_constants) * 2.
        assert np.allclose(hess, hess_rebuilt)
    print('\tOK!')

    print('\nCHECKING MINAMIDE DECOMPOSITION')
    for i in range(10):
        xy = np.random.randn(n+m)

        # system is (S + phi phi^T)

        # pseudo-solve the full system
        hess = my_hessian(xy)
        hessian_pinv = np.linalg.pinv(hess) #.A

        # build the reduced system
        hess_x_ng = my_hessian_x_nogap(xy[:n])
        hess_y_ng = my_hessian_y_nogap(xy[n:])
        S = np.bmat([
            [hess_x_ng, np.zeros((n, m))],
            [np.zeros((m, n)), hess_y_ng]]).A
        phi = np.concatenate([c, b]) * np.sqrt(2.)

        # obtain pinv of the reduced system
        Splus = np.linalg.pinv(S)

        # build T matrix (projector on null space of S)
        T = np.eye(n+m) - Splus @ S

        assert np.allclose(T, np.eye(n+m) - S @ Splus)

        # get part of phi in the nullspace
        Tphi = T @ phi

        # simple case, ~Woodbury
        if np.allclose(Tphi, 0.):
            print('Case 2: phi orthogonal to the null space')

            # example 2.4 of paper, case 2.: Woodbury formula with pinv's
            Splusphi = Splus @ phi
            pinv_rebuilt = Splus - np.outer(Splusphi, Splusphi) / (
                1 + phi @ Splusphi)

        # interesting case
        else:
            print('Case 1: phi not orthogonal to the null space')

            # example 2.4 of paper, case 1.
            right = np.eye(n+m) - np.outer(phi, Tphi) / (phi @ Tphi)
            left = right.T
            extra = np.outer(Tphi, Tphi) / ((phi @ Tphi)**2)
            pinv_rebuilt = left @ Splus @ right + extra

        # main goal
        assert np.allclose(hessian_pinv, pinv_rebuilt)

    print('\tOK!')
