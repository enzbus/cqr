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
"""Solver class.

Idea:

Centralizes memory allocation, its managed memory translates to a struct in C.
Each method, which should be very small and simple, translates to a C function.
Experiments (new features, ...) should be done as subclasses.
"""

# import cvxpy as cp
import numpy as np
import scipy as sp

class Unbounded(Exception):
    """Program unbounded."""

class Infeasible(Exception):
    """Program infeasible."""

class Solver:
    """Solver class.

    :param matrix: Problem data matrix.
    :type n: sp.sparse.csc_matrix
    :param b: Dual cost vector.
    :type b: np.array
    :param c: Primal cost vector.
    :type c: np.array
    :param zero: Size of the zero cone.
    :type zero: int
    :param nonneg: Size of the non-negative cone.
    :type nonneg: int
    :param x0: Initial guess of the primal variable. Default None, equivalent
        to zero vector.
    :type x0: np.array or None.
    :param y0: Initial guess of the dual variable. Default None, equivalent
        to zero vector.
    :type y0: np.array or None.
    """

    def __init__(self, matrix, b, c, zero, nonneg, x0=None, y0=None):
        
        # process program data
        self.matrix = sp.sparse.csc_matrix(matrix)
        self.m, self.n = matrix.shape
        assert zero >= 0
        assert nonneg >= 0
        assert zero + nonneg == self.m
        self.zero = zero
        self.nonneg = nonneg
        assert len(b) == self.m
        self.b = np.array(b, dtype=float)
        assert len(c) == self.n
        self.c = np.array(c, dtype=float)

        # process initial guess
        self.x = np.empty(self.n, dtype=float)
        self.y = np.empty(self.m, dtype=float)
        self.update_variables(x0=x0, y0=y0)

        self._qr_transform_program_data()
        self._qr_transform_dual_space()
        self._qr_transform_gap()

        # self.toy_solve()
        self.new_toy_solve()
        # self.x_transf, self.y = self.solve_program_cvxpy(
        #     self.matrix_qr_transf, b, self.c_qr_transf)
        self._invert_qr_transform_dual_space()
        self.invert_qr_transform()


    def backsolve_r(self, vector):
        """Simple triangular solve with matrix R."""
        result = np.linalg.lstsq(self.r.T, vector, rcond=None)[0]
        if not np.allclose(self.r.T @ result, vector):
            raise Unbounded(
                "Cost vector is not in the span of the program matrix!")
        return result
        
    def update_variables(self, x0=None, y0=None):
        """Update initial values of the primal and dual variables.

        :param x0: Initial guess of the primal variable. Default None,
            equivalent to zero vector.
        :type x0: np.array or None.
        :param y0: Initial guess of the dual variable. Default None, equivalent
            to zero vector.
        :type y0: np.array or None.
        """

        if x0 is None:
            self.x[:] = np.zeros(self.n, dtype=float)
        else:
            assert len(x0) == self.n
            self.x[:] = np.array(x0, dtype=float)
        if y0 is None:
            self.y[:] = np.zeros(self.m, dtype=float)
        else:
            assert len(y0) == self.m
            self.y[:] = np.array(y0, dtype=float)

    # @staticmethod
    # def solve_program_cvxpy(A, b, c):
    #     """Solve simple LP with CVXPY."""
    #     m, n = A.shape
    #     x = cp.Variable(n)
    #     constr = [b - A @ x >= 0]
    #     cp.Problem(cp.Minimize(x.T @ c), constr).solve()
    #     return x.value, constr[0].dual_value


    def _qr_transform_program_data(self):
        """Apply QR decomposition to equilibrated program data."""
        # assert self.m > self.n, "Case m <= n not yet implemented."

        q, r = np.linalg.qr(self.matrix.todense(), mode='complete')
        print('diagonal of R')
        print(np.diag(r))
        self.matrix_qr_transf = q[:, :self.n].A
        self.nullspace_projector = q[:, self.n:].A
        self.r = r[:self.n].A
        # breakpoint()

        print('c')
        print(self.c)
        self.c_qr_transf = self.backsolve_r(self.c)
        print('c transf')
        print(self.c_qr_transf)
        # breakpoint()

        # self.sigma_qr = np.linalg.norm(
        #     self.b) #/ np.mean(np.linalg.norm(matrix_transf, axis=1))
        # self.b_qr_transf = self.b/self.sigma_qr

    def invert_qr_transform(self):
        """Simple triangular solve with matrix R."""
        result = np.linalg.lstsq(self.r, self.x_transf, rcond=None)[0]
        if not np.allclose(self.r @ result, self.x_transf):
            raise Exception
            #raise Unbounded(
            #    "Cost vector is not in the span of the program matrix!")
        self.x = result

    def _qr_transform_dual_space(self):
        """Apply QR transformation to dual space."""
        self.y0 = self.matrix_qr_transf @ -self.c_qr_transf
        if self.m <= self.n:
            if not np.all(self.y0 >= 1e-12):
                raise Unbounded("There is no feasible dual vector.")
        # diff = self.y - self.y0
        # self.y_reduced = self.nullspace_projector.T @ diff
        self.b0 = self.b @ self.y0
        self.b_reduced = self.b @ self.nullspace_projector

    def _invert_qr_transform_dual_space(self):
        """Apply QR transformation to dual space."""
        self.y = self.y0 + self.nullspace_projector @ self.y_reduced

    def _qr_transform_gap(self):
        """Apply QR transformation to zero-gap residual."""
        Q, R = np.linalg.qr(np.concatenate(
            [self.c_qr_transf, self.b_reduced]).reshape((self.m,1)), mode='complete')
        self.gap_NS = Q[:, 1:]
        self.var0 = - self.b0 * np.concatenate(
            [self.c_qr_transf, self.b_reduced]) / np.linalg.norm(np.concatenate([self.c_qr_transf, self.b_reduced]))**2

    def newres(self, var_reduced):
        """Residual using gap QR transform."""
        var = self.var0 + self.gap_NS @ var_reduced
        x = var[:self.n]
        y_reduced = var[self.n:]
        if self.m <= self.n:
            return self.pri_res(x)
        return np.concatenate(
            [self.pri_res(x), self.dua_res(y_reduced)])

    def newjacobian(self, var_reduced):
        var = self.var0 + self.gap_NS @ var_reduced
        x = var[:self.n]
        y_reduced = var[self.n:]
        s_active = 1. * ((self.b - self.matrix_qr_transf @ x) < 0.)
        y_active = 1. * ((self.y0 + self.nullspace_projector @ y_reduced) < 0.)
        if self.m <= self.n:
            result = np.block(
                [[-np.diag(s_active) @ self.matrix_qr_transf]])
        else:
            result = np.block(
                [[-np.diag(s_active) @ self.matrix_qr_transf, np.zeros((self.m, self.m-self.n))],
                [np.zeros((self.m, self.n)), np.diag(y_active) @ self.nullspace_projector],
                ])    
        return result @ self.gap_NS

    def pri_res(self, x):
        return np.minimum(self.b - self.matrix_qr_transf @ x, 0.)

    def dua_res(self, y_reduced):
        return np.minimum(self.y0 + self.nullspace_projector @ y_reduced, 0.)

    def gap(self, x, y_reduced):
        return self.c_qr_transf.T @ x + self.b_reduced @ y_reduced + self.b0

    def res(self, var):
        x = var[:self.n]
        y_reduced = var[self.n:]
        if self.m <= self.n:
            return np.concatenate(
                [self.pri_res(x), [self.gap(x,y_reduced)]])
        return np.concatenate(
            [self.pri_res(x), self.dua_res(y_reduced), [self.gap(x,y_reduced)]])

    def jacobian(self, var):
        x = var[:self.n]
        y_reduced = var[self.n:]
        s_active = 1. * ((self.b - self.matrix_qr_transf @ x) < 0.)
        y_active = 1. * ((self.y0 + self.nullspace_projector @ y_reduced) < 0.)
        if self.m <= self.n:
            return np.block(
                [[-np.diag(s_active) @ self.matrix_qr_transf],
                [self.c_qr_transf.reshape(1,self.m)]])
        return np.block(
            [[-np.diag(s_active) @ self.matrix_qr_transf, np.zeros((self.m, self.m-self.n))],
            [np.zeros((self.m, self.n)), np.diag(y_active) @ self.nullspace_projector],
            [self.c_qr_transf.reshape(1,self.n), self.b_reduced.reshape(1, self.m-self.n)] 
            ])    

    def toy_solve(self):
        result = sp.optimize.least_squares(
            self.res, np.zeros(self.m),
            jac=self.jacobian, method='lm')
        var = result.x
        self.x_transf = var[:self.n]
        self.y_reduced = var[self.n:]

    def new_toy_solve(self):
        result = sp.optimize.least_squares(
            self.newres, np.zeros(self.m-1),
            jac=self.newjacobian, method='lm')
        print(result)
        
        if result.cost > 1e-12:
            # infeasible; for convenience we just set this here,
            # will have to check which is valid and maybe throw exceptions
            self.y = -self.newres(result.x)[:self.m]
            if np.linalg.norm(self.y)**2 > 1e-12:
                print('infeasibility certificate')
                print(self.y)
                raise Infeasible()

            s_certificate = -self.newres(result.x)[self.m:]
            if np.linalg.norm(s_certificate)**2 > 1e-12:
                print('unboundedness certificate')
                self.x_transf = - self.matrix_qr_transf.T @ s_certificate
                self.invert_qr_transform()
                raise Unbounded()

            # breakpoint()

            # var = self.var0 + self.gap_NS @ result.x
            # y_reduced = var[self.n:]
            # y = self.y0 + self.nullspace_projector @ y_reduced
            # elf.unboundedness_certificate = - (self.matrix.T @ y + self.c)
            
            # self.invert_qr_transform()

            # assert np.min(self.infeasibility_certificate) >= -1e-6
            # assert np.allclose(self.matrix.T @ self.infeasibility_certificate, 0.)
            # assert self.b.T @ self.infeasibility_certificate < 0.

        var_reduced = result.x
        var = self.var0 + self.gap_NS @ var_reduced
        self.x_transf = var[:self.n]
        self.y_reduced = var[self.n:]

        