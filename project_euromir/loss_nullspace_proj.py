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
"""Loss and related functions for nullspace projection model."""

import numpy as np
import scipy as sp

from .direction_calculator import DirectionCalculator


def _densify_also_nonsquare(linear_operator):
    result = np.empty(linear_operator.shape)
    for j in range(linear_operator.shape[1]):
        e_j = np.zeros(linear_operator.shape[1])
        e_j[j] = 1.
        result[:, j] = linear_operator.matvec(e_j)
    return result


class NullSpaceModel(DirectionCalculator):

    def __init__(
        self, m, n, zero, nonneg, matrix_transfqr, b, c, nullspace_projector):
        self.m = m
        self.n = n
        self.zero = zero
        self.nonneg = nonneg
        self.matrix = matrix_transfqr
        self.b = b
        self.c = c
        self.nullspace_projector = nullspace_projector
        self.b_proj = self.b @ self.nullspace_projector

        # since matrix is from QR
        self.y0 = -self.c @ self.matrix.T

    def loss(self, variable):
        x = variable[:self.n]
        y_null = variable[self.n:]
        y = self.y0 + self.nullspace_projector @ y_null
        s = -self.matrix @ x + self.b
        y_loss = np.linalg.norm(np.minimum(y[self.zero:], 0.))**2
        s_loss_zero = np.linalg.norm(s[:self.zero])**2
        s_loss_nonneg = np.linalg.norm(np.minimum(s[self.zero:], 0.))**2
        gap_loss = (self.c.T @ x + self.b.T @ y)**2
        return (y_loss + s_loss_zero + s_loss_nonneg + gap_loss
            ) / 2.

    def gradient(self, variable):
        x = variable[:self.n]
        y_null = variable[self.n:]
        y = self.y0 + self.nullspace_projector @ y_null
        s = -self.matrix @ x + self.b
        y_error = np.minimum(y[self.zero:self.zero+self.nonneg], 0.)
        s_error = np.copy(s)
        s_error[self.zero:self.zero+self.nonneg] = np.minimum(
            s[self.zero:self.zero+self.nonneg], 0.)
        gap = self.c.T @ x + self.b.T @ y

        gradient = np.empty_like(variable)
        gradient[:self.n] = -self.matrix.T @ s_error
        gradient[self.n:] = self.nullspace_projector[self.zero:].T @ y_error

        gradient[:self.n] += gap * self.c
        gradient[self.n:] += gap * self.b_proj
        return gradient

    def hessian(self, variable, regularizer = None):
        x = variable[:self.n]
        y_null = variable[self.n:]
        y = self.y0 + self.nullspace_projector @ y_null
        s = -self.matrix @ x + self.b
        y_error = np.minimum(y[self.zero:self.zero+self.nonneg], 0.)
        s_error = np.copy(s)
        s_error[self.zero:self.zero+self.nonneg] = np.minimum(
            s[self.zero:self.zero+self.nonneg], 0.)
        gap = self.c.T @ x + self.b.T @ y

        # this is DProj
        s_mask = np.ones(self.m, dtype=float)
        s_mask[self.zero:] = s_error[self.zero:] < 0.
        y_mask = np.zeros(self.m, dtype=float)
        y_mask[self.zero:] = y_error < 0.

        if regularizer is not None:
            s_mask += regularizer
            y_mask += regularizer
            # if there is no regularizer, the masks have values in {0, 1},
            # so there is no need to square them, but with reg they have
            # different values, so we square them
            s_mask = s_mask ** 2
            y_mask = y_mask ** 2

        def _matvec(dvar):
            result = np.empty_like(dvar)
            dx = dvar[:self.n]
            dy_null = dvar[self.n:]

            # s_error sqnorm
            result[:self.n] = (self.matrix.T @ (s_mask * (self.matrix @ dx)))

            # y_error sqnorm
            result[self.n:] = (self.nullspace_projector.T @ (
                y_mask * (self.nullspace_projector @ dy_null)))

            # gap
            constants = np.concatenate([self.c, self.b_proj])
            result[:] += constants * (constants @ dvar)

            return result # + regularizer * dxdy

        return sp.sparse.linalg.LinearOperator(
            shape=(len(variable), len(variable)),
            matvec=_matvec
        )

    def dense_hessian(self, variable, **kwargs):
        return _densify_also_nonsquare(
            self.hessian(variable=variable, **kwargs))

    def get_direction(
        self, current_point: np.array, current_gradient: np.array) -> np.array:
        """Exact matrix solve."""

        x = current_point[:self.n]
        y_null = current_point[self.n:]
        y = self.y0 + self.nullspace_projector @ y_null
        s = -self.matrix @ x + self.b
        y_error = np.minimum(y[self.zero:self.zero+self.nonneg], 0.)
        s_error = np.copy(s)
        s_error[self.zero:self.zero+self.nonneg] = np.minimum(
            s[self.zero:self.zero+self.nonneg], 0.)
        gap = self.c.T @ x + self.b.T @ y
        constants = np.concatenate([self.c, self.b_proj])

        # this is DProj
        s_mask = np.ones(self.m, dtype=float)
        s_mask[self.zero:] = s_error[self.zero:] < 0.
        y_mask = np.zeros(self.m, dtype=float)
        y_mask[self.zero:] = y_error < 0.

        # simple
        regularizer = np.linalg.norm(current_gradient)

        s_mask += regularizer
        y_mask += regularizer
        # if there is no regularizer, the masks have values in {0, 1},
        # so there is no need to square them, but with reg they have
        # different values, so we square them
        s_mask = s_mask ** 2
        y_mask = y_mask ** 2

        def base_solver(dvar):
            """Solve by the block diag part."""
            result = np.empty_like(dvar)
            dx = dvar[:self.n]
            dy_null = dvar[self.n:]

            # s_error sqnorm
            result[:self.n] = (self.matrix.T @ (
                (1./s_mask) * (self.matrix @ dx)))

            # y_error sqnorm
            result[self.n:] = (self.nullspace_projector.T @ (
                (1./y_mask) * (self.nullspace_projector @ dy_null)))
            
            return result

        # Sherman-Morrison
        Ainvu = base_solver(constants)
        denom = 1 + constants @ Ainvu

        result = base_solver(current_gradient) - Ainvu * (
            Ainvu @ current_gradient) / denom 
        
        return -result



if __name__ == '__main__': # pragma: no cover

    from scipy.optimize import check_grad

    # create consts
    np.random.seed(0)
    m = 20
    n = 10
    zero = 5
    nonneg = 15
    matrix = np.random.randn(m, n)
    q, r = np.linalg.qr(matrix, mode='complete')
    matrix_transfqr = q[:, :n]
    nullspace_proj = q[:, n:]

    b = np.random.randn(m)
    c = np.random.randn(n)

    model = NullSpaceModel(
        m=m, n=n, zero=zero, nonneg=nonneg, matrix_transfqr=matrix_transfqr,
        b=b, c=c, nullspace_projector=nullspace_proj)

    print('\nCHECKING GRADIENT')
    for i in range(10):
        print(check_grad(model.loss, model.gradient, np.random.randn(m)))

    print('\nCHECKING HESSIAN')
    for i in range(10):
        print(check_grad(model.gradient, model.dense_hessian,
            np.random.randn(m)))

    print('\nCHECKING SOLVE')
    for i in range(10):
        var = np.random.randn(m)
        grad = model.gradient(var)
        H = model.dense_hessian(var, regularizer=np.linalg.norm(grad))
        result = np.linalg.solve(H, - grad)
        result1 = model.get_direction(var, grad)
        print(np.linalg.norm(result - result1) / np.linalg.norm(result + result1))

