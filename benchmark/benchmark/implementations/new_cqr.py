# Copyright 2024 Enzo Busseti
#
# This file is part of CQR, the Conic QR Solver.
#
# CQR is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# CQR is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# CQR. If not, see <https://www.gnu.org/licenses/>.
"""New CQR idea, ADMM based."""

import numpy as np
import scipy as sp

from ..base_solver import BaseSolver

from pyspqr import qr

NUMPY = True
PYSPQR = False
class NewCQR(BaseSolver):
    """New idea for base CQR formulation."""

    max_iterations = 100000

    used_matrix = "matrix"
    used_b = "b"
    used_c = "c"

    def prepare_loop(self):
        """Define anything we need to re-use."""

        matrix = getattr(self, self.used_matrix)

        if NUMPY:
            q, r = np.linalg.qr(
                getattr(self, self.used_matrix).todense(), mode='complete')
            self.qr_matrix = q[:, :self.n].A
            self.nullspace = q[:, self.n:].A
            self.triangular = r[:self.n].A
        if PYSPQR:
            matrix.indices = matrix.indices.astype(np.int32)
            matrix.indptr = matrix.indptr.astype(np.int32)
            q, r, e = qr(matrix, ordering='AMD')
            shape1 = min(self.n, self.m)
            self.qr_matrix = sp.sparse.linalg.LinearOperator(
                shape=(self.m, shape1),
                matvec=lambda x: q @ np.concatenate([x, np.zeros(self.m-shape1)]),
                rmatvec=lambda y: (
                    q.T @ np.array(y, copy=True).reshape(y.size))[:shape1],
            )
            shape2 = max(self.m - self.n, 0)
            self.nullspace = sp.sparse.linalg.LinearOperator(
                shape=(self.m, shape2),
                matvec=lambda x: q @ np.concatenate([np.zeros(self.m-shape2), x]),
                rmatvec=lambda y: (
                    q.T @ np.array(y, copy=True).reshape(y.size))[self.m-shape2:]
            )
            self.triangular = sp.sparse.csc_array((r.todense() @ e)[:self.n])
            # of course this is inefficient
            self.triangular_solve = sp.sparse.linalg.splu(self.triangular)
            self.triangular_solve_transpose = sp.sparse.linalg.splu(self.triangular.T)

        # assert np.allclose(
        #     self.qr_matrix @ self.triangular, matrix.todense())

        if NUMPY:
            self.c_qr = sp.linalg.solve_triangular(
                self.triangular.T, getattr(self, self.used_c), lower=True)
        if PYSPQR:
            self.c_qr = self.triangular_solve_transpose.solve(
                getattr(self, self.used_c))

        # shift in the linspace projector
        self.e = self.nullspace @ self.nullspace.T @ (
            self.qr_matrix @ self.c_qr - getattr(self, self.used_b)
                ) - self.qr_matrix @ self.c_qr

        # # shift in the linspace projector
        # self.e = (self.nullspace @ self.nullspace.T) @ (
        #     matrix @ self.c - self.b) - matrix @ self.c

        self.z = np.zeros(self.m)
        self.y = np.zeros(self.m)
        self.s = np.zeros(self.m)
        self.x = np.zeros(self.n)

    def cone_project(self, z):
        """Project on y cone."""
        return self.composed_cone_project(
            z, has_zero=False, has_free=True, has_hsde=False)

    def linspace_project(self, y_plus_s):
        """Linspace project (y+s) -> y."""
        return self.nullspace @ (self.nullspace.T @ y_plus_s) + self.e

    def iterate(self):
        """Simple Douglas Rachford iteration."""
        self.y[:] = self.cone_project(self.z)
        step = self.linspace_project(2 * self.y - self.z) - self.y
        # print(np.linalg.norm(step))
        self.z[:] = self.z + step
        # self.y[:] = self.cone_project(self.z)

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        self.y[:] = self.cone_project(self.z)
        self.s[:] = self.y - self.z
        x_qr = self.qr_matrix.T @ (getattr(self, self.used_b) - self.s)
        # breakpoint()
        if NUMPY:
            self.x[:] = sp.linalg.solve_triangular(self.triangular, x_qr, lower=False)
        if PYSPQR:
            self.x[:] = self.triangular_solve.solve(x_qr)
        # breakpoint()


class LevMarNewCQR(NewCQR):
    """Using Levemberg Marquardt."""

    lsqr_iters = 5
    max_iterations = 100000//(2 * lsqr_iters + 1)

    def multiply_cone_project_derivative(self, z, dz):
        """Derivative projection on y cone."""

        result = np.zeros_like(z)

        # zero cone
        result[:self.zero] = dz[:self.zero]
        cur = self.zero

        # nonneg cone
        result[cur:cur+self.nonneg] = (
            z[cur:cur+self.nonneg] > 0.) * dz[cur:cur+self.nonneg]
        cur += self.nonneg

        # soc cones
        for soc_dim in self.soc:
            result[cur:cur+soc_dim] = \
                self.multiply_jacobian_second_order_project(
                    z[cur:cur+soc_dim], dz[cur:cur+soc_dim])
            cur += soc_dim
        assert cur == self.m

        return result

    def iterate(self):
        """Do one iteration."""

        self.y[:] = self.cone_project(self.z)
        step = self.linspace_project(2 * self.y - self.z) - self.y

        result = sp.sparse.linalg.lsqr(
            sp.sparse.linalg.LinearOperator(
                shape=(self.m, self.m),
                matvec=lambda dz: self.multiply_jacobian_dstep(self.z, dz),
                rmatvec=lambda dr: self.multiply_jacobian_dstep_transpose(
                    self.z, dr)), -step,
                    x0=step,
                    damp=0.0, # might make sense to change this?
                    atol=0., btol=0., # might make sense to change this
                    iter_lim=self.lsqr_iters)
        # breakpoint()
        # print(result[1:-1])
        self.z[:] = self.z + result[0]

    def linspace_project_derivative(self, dz):
        """Derivative linspace project (y+s) -> y."""
        return self.nullspace @ (self.nullspace.T @ dz)

    def multiply_jacobian_dstep(self, z, dz):
        """Multiply by Jacobian of DR step operator."""
        # breakpoint()
        tmp = self.multiply_cone_project_derivative(z, dz)
        return self.linspace_project_derivative(2 * tmp - dz) - tmp

    def multiply_jacobian_dstep_transpose(self, z, dr):
        """Multiply by Jacobian of DR step operator transpose."""
        tmp = self.linspace_project_derivative(dr)
        return self.multiply_cone_project_derivative(z, 2 * tmp - dr) - tmp


class EquilibratedNewCQR(NewCQR):
    """With Ruiz equilibration."""

    # max_iterations = 1000

    used_matrix = "eq_matrix"
    used_b = "eq_b"
    used_c = "eq_c"

    def prepare_loop(self):
        """Do Ruiz equilibration."""
        if len(self.soc) > 0:
            raise ValueError()
        matrix = self.matrix.todense()
        concatenated = np.block(
            [[matrix, self.b.reshape(self.m, 1)],
            [self.c.reshape(1, self.n), np.zeros((1, 1))]]).A
        work_matrix = np.copy(concatenated)

        def norm_cols(concatenated):
            return np.max(np.abs(concatenated), axis=0)

        def norm_rows(concatenated):
            return np.max(np.abs(concatenated), axis=1)

        m, n = matrix.shape

        d_and_rho = np.ones(m+1)
        e_and_sigma = np.ones(n+1)

        for i in range(100):

            nr = norm_rows(work_matrix)
            nc = norm_cols(work_matrix)

            r1 = max(nr[nr > 0]) / min(nr[nr > 0])
            r2 = max(nc[nc > 0]) / min(nc[nc > 0])
            # print(r1, r2)
            if (r1-1 < 1e-5) and (r2-1 < 1e-5):
                # logger.info('Equilibration converged.')
                break

            # print(r1, r2)

            d_and_rho[nr > 0] *= nr[nr > 0]**(-0.5)
            e_and_sigma[nc > 0] *= ((m+1)/(n+1))**(0.25) * nc[nc > 0]**(-0.5)

            work_matrix = ((concatenated * e_and_sigma).T * d_and_rho).T

        self.equil_e = e_and_sigma[:-1]
        self.equil_d = d_and_rho[:-1]
        self.equil_sigma = e_and_sigma[-1]
        self.equil_rho = d_and_rho[-1]

        self.eq_matrix = sp.sparse.csc_matrix(work_matrix[:-1, :-1])
        self.eq_b = work_matrix[:-1, -1]
        self.eq_c = work_matrix[-1, :-1]

        super().prepare_loop()

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        super().obtain_x_and_y()

        self.x = (self.equil_e * self.x) / self.equil_sigma
        self.y = (self.equil_d * self.y) / self.equil_rho
