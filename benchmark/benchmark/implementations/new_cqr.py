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

        # self.allsteps = []

    def cone_project(self, z):
        """Project on y cone."""
        return self.composed_cone_project(
            z, has_zero=False, has_free=True, has_hsde=False)

    def linspace_project(self, y_plus_s):
        """Linspace project (y+s) -> y."""
        return self.nullspace @ (self.nullspace.T @ y_plus_s) + self.e

    # def dr_step(self, z):
    #     """DR step."""
    #     y = self.cone_project(z)
    #     return self.linspace_project(2 * y - z) - y

    def iterate(self):
        """Simple Douglas Rachford iteration."""
        self.y[:] = self.cone_project(self.z)
        step = self.linspace_project(2 * self.y - self.z) - self.y
        # print(np.linalg.norm(step))
        self.z[:] = self.z + step
        # self.y[:] = self.cone_project(self.z)

        # self.allsteps.append(np.copy(step))

        # if len(self.solution_qualities) > 50000:
        #     import matplotlib.pyplot as plt
        #     breakpoint()

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

# def data_ql_transform(A: np.array, b: np.array, c: np.array):
#     """Prototype using numpy."""
#     n = len(c)
#     m = len(b)
#     if m < n:
#         raise NotImplementedError("Not implemented case n>m yet.")
#     matrix = np.block([
#         [np.zeros((1, 1)), c.reshape(1, n)],
#         [-b.reshape(m, 1), A],
#     ])
#     _q, _r = np.linalg.qr(matrix[::-1, ::-1], mode='complete')
#     q = _q[::-1, ::-1].A
#     l = _r[::-1, ::-1].A

#     # remove zeros
#     l = l[-n-1:]

#     # scale
#     scale = l[0, 0] # need guard here if 0; when does that happen?
#     # assert scale > 0
#     l /= scale
#     q_scaled = q * scale

#     # split for nullspace
#     q_matrix = q_scaled[:, -n-1:]
#     q_nullspace = q_scaled[:, :-n-1]

#     assert np.allclose(
#         sp.linalg.solve_triangular(l.T, matrix.T, lower=False), q_matrix.T)
#     assert np.isclose(l[0, 0], 1.)
#     A_transf = q_matrix[1:, 1:]
#     c_transf = q_matrix[0, 1:]
#     b_transf = -q_matrix[1:, 0]
#     return A_transf, c_transf, b_transf, (q, scale), l

# class HSDENewCQR(BaseSolver):
#     """Same as NewCQR but using HSDE, nasty formulas.

#     Huge notebook with derivation, not in git, but I guess can be derived
#     much simpler using (I + Q)^-1 formulation.

#     We do Ac QR decomp, doing Abc QL decomp seems to give no advantage.
#     """

#     used_matrix = "matrix"
#     used_b = "b"
#     used_c = "c"

#     max_iterations = 100

#     def prepare_loop(self):
#         """Define anything we need to re-use."""

#         matrix = getattr(self, self.used_matrix).todense()
#         b = getattr(self, self.used_b)
#         c = getattr(self, self.used_c)

#         Aqr_, c_qr, b_qr, (q,self.scale), l = data_ql_transform(matrix, b, c)
#         self.q_ac = q[:,-self.n:]
#         self.q_nullspace = q[:,:-self.n]

#         self.pseudo_householder = np.block([
#             [np.zeros((1,1)), -b_qr.reshape(1,self.m)],
#             [b_qr.reshape(self.m,1), np.zeros((self.m,self.m))],])

#         self.explicit_linspace_projector = (
#             np.linalg.inv(np.eye(self.m+1) + self.q_nullspace @ self.q_nullspace.T @ self.pseudo_householder)
#             @ (self.q_nullspace @ self.q_nullspace.T))

#         # matrix_to_decompose = np.block( # shape (m+1, n)
#         #     [[c.reshape(1,self.n)], [matrix]])
#         # q, r = np.linalg.qr(matrix_to_decompose, mode='complete')
#         # self.qr_matrix = q[:, :self.n].A # shape (m+1, n)
#         # self.nullspace = q[:, self.n:].A # shape (m+1, m+1-n)
#         # self.triangular = r[:self.n].A # shape (n, n)

# #         self.pseudo_householder = np.block([
# #             [np.zeros((1,1)), -b.reshape(1,self.m)],
# #     [       +b.reshape(self.m,1), np.zeros((self.m,self.m))],
# # ])

# #         self.explicit_linspace_projector = (
# #             np.linalg.inv(np.eye(self.m+1) + self.nullspace @ self.nullspace.T @ self.pseudo_householder)
# #                 @ (self.nullspace @ self.nullspace.T))

#         self.z = np.zeros(self.m+1)
#         self.z[0] = 1.
#         self.u_ykappa = np.copy(self.z)
#         self.v_stau = np.zeros_like(self.z)
#         self.u_x = np.zeros(self.n)

#     def cone_project(self, z):
#         """Project on (y, kappa) cone."""
#         return self.composed_cone_project(
#             z, has_zero=False, has_free=True, has_hsde=False, has_hsde_first=True,)

#     def iterate(self):
#         """Simple Douglas Rachford iteration."""
#         self.u_ykappa[:] = self.cone_project(self.z)
#         step = self.explicit_linspace_projector @ (2 * self.u_ykappa - self.z) - self.u_ykappa
#         print(np.linalg.norm(step), self.z[0])
#         self.z[:] = self.z + step

#     def obtain_x_and_y(self):
#         """Redefine if/as needed."""
#         self.u_ykappa[:] = self.cone_project(self.z)
#         self.v_stau[:] = self.u_ykappa - self.z
#         # breakpoint()
#         u_x = self.q_ac.T @ ((self.pseudo_householder @ self.u_ykappa - self.v_stau)/self.scale)
#         breakpoint()
#     #     self.u_x[:] = self.qr_matrix.T @ (self.pseudo_householder @ self.u_ykappa - self.v_stau)
#         # assert np.allclose((-self.scale) * self.q_ac @ u_x + self.pseudo_householder @ self.u_ykappa, self.v_stau)
#     #     # breakpoint()
#     #     kappa = self.u_ykappa[0]
#     #     if kappa == 0:
#     #         return
#     #     self.u_x /= np.abs(kappa)
#     #     self.y[:] = self.u_ykappa[1:] / np.abs(kappa)
#     #     self.x[:] = sp.linalg.solve_triangular(self.triangular, self.u_x, lower=False)


class LevMarNewCQR(NewCQR):
    """Using Levemberg Marquardt."""

    lsqr_iters = 5
    max_iterations = 100000//(2 * lsqr_iters + 1)
    damp = 0.

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
        # print(np.linalg.norm(step))

        result = sp.sparse.linalg.lsqr(
            sp.sparse.linalg.LinearOperator(
                shape=(self.m, self.m),
                matvec=lambda dz: self.multiply_jacobian_dstep(self.z, dz),
                rmatvec=lambda dr: self.multiply_jacobian_dstep_transpose(
                    self.z, dr)), -step,
                    x0=step,
                    damp=self.damp, # might make sense to change this?
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

class LevMarUnitDampNewCQR(LevMarNewCQR):
    """Using Levemberg Marquardt."""
    damp = 1.0

# class LongNewCQR(LevMarNewCQR):
#     """Test with long LSQR loop."""

#     max_iterations = 10000

#     def prepare_loop(self):
#         """Skip SOCs."""
#         assert len(self.soc) == 0
#         super().prepare_loop()

#         mat = self.nullspace @ self.nullspace.T
#         mat[np.abs(mat) < np.finfo(float).eps] = 0.
#         self.base_mat = sp.sparse.csc_array(mat)

#     def iterate(self):
#         """Do one iteration."""

#         self.y[:] = self.cone_project(self.z)
#         step = self.linspace_project(2 * self.y - self.z) - self.y
#         # print(np.linalg.norm(step))
#         # if len(self.solution_qualities) < 10:
#         #     self.z[:] = self.z + step
#         #     return

#         mask = np.ones(self.m)
#         mask[self.zero:] = self.z[self.zero:] > 0
#         actual_mat = self.base_mat @ sp.sparse.diags(2 * mask - 1.) - sp.sparse.diags(mask)
#         regularizer = 0.1
#         actual_mat = actual_mat * (1 - regularizer) + regularizer * (-1) * sp.sparse.eye(self.m)
#         newstep =  sp.sparse.linalg.spsolve(actual_mat, -step)
#         self.z[:] = self.z + newstep

#         # result = sp.sparse.linalg.lsqr(
#         #     sp.sparse.linalg.LinearOperator(
#         #         shape=(self.m, self.m),
#         #         matvec=lambda dz: self.multiply_jacobian_dstep(self.z, dz),
#         #         rmatvec=lambda dr: self.multiply_jacobian_dstep_transpose(
#         #             self.z, dr)), -step,
#         #             x0=step,
#         #             damp=1e-2, # might make sense to change this?
#         #             atol=0., btol=0., # might make sense to change this
#         #             iter_lim=self.m)
#         # # breakpoint()
#         # # print(result[1:-1])
#         # self.z[:] = self.z + result[0]


class EquilibratedNewCQR(NewCQR):
    """With Ruiz equilibration."""

    # max_iterations = 1000

    used_matrix = "eq_matrix"
    used_b = "eq_b"
    used_c = "eq_c"

    def prepare_loop(self):
        """Do Ruiz equilibration."""
        # if len(self.soc) > 0:
        #     raise ValueError()
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

            # equalize nr for SOCs
            cur = self.zero + self.nonneg
            for soc_dim in self.soc:
                nr[cur:cur+soc_dim] = np.max(nr[cur:cur+soc_dim])
                cur += soc_dim
            # breakpoint()

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

class L2EquilibratedNewCQR(EquilibratedNewCQR):
    """Same but with l2 equilibration."""

    def prepare_loop(self):
        """Do Ruiz equilibration."""
        # if len(self.soc) > 0:
        #     raise ValueError()
        matrix = self.matrix.todense()
        concatenated = np.block(
            [[matrix, self.b.reshape(self.m, 1)],
            [self.c.reshape(1, self.n), np.zeros((1, 1))]]).A
        work_matrix = np.copy(concatenated)

        def norm_cols(concatenated):
            return np.linalg.norm(concatenated, axis=0)

        def norm_rows(concatenated):
            return np.linalg.norm(concatenated, axis=1)

        m, n = matrix.shape

        d_and_rho = np.ones(m+1)
        e_and_sigma = np.ones(n+1)

        for i in range(100):

            nr = norm_rows(work_matrix)
            nc = norm_cols(work_matrix)

            # equalize nr for SOCs
            cur = self.zero + self.nonneg
            for soc_dim in self.soc:
                nr[cur:cur+soc_dim] = np.linalg.norm(
                    nr[cur:cur+soc_dim])/np.sqrt(soc_dim)
                cur += soc_dim
            # breakpoint()

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

        NewCQR.prepare_loop(self)

class EquilibratedLevMarNewCQR(EquilibratedNewCQR, LevMarNewCQR):
    """Equilibrated Lev Mar."""
