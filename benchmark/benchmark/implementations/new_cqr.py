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
    use_numpy = True

    def prepare_loop(self):
        """Define anything we need to re-use."""

        matrix = getattr(self, self.used_matrix)

        if self.use_numpy:
            q, r = np.linalg.qr(
                getattr(self, self.used_matrix).todense(), mode='complete')
            self.qr_matrix = q[:, :self.n].A
            self.nullspace = q[:, self.n:].A
            self.triangular = r[:self.n].A
        else:
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
            self.pyspqr_r = r[:self.n]
            self.pyspqr_e = e
            self.triangular = sp.sparse.csc_array((r.todense() @ e)[:self.n])
            # of course this is inefficient
            # self.triangular_solve = sp.sparse.linalg.splu(self.triangular)
            # self.triangular_solve_transpose = sp.sparse.linalg.splu(self.triangular.T)

        # assert np.allclose(
        #     self.qr_matrix @ self.triangular, matrix.todense())

        if self.use_numpy:
            self.c_qr = sp.linalg.solve_triangular(
                self.triangular.T, getattr(self, self.used_c), lower=True)
        else:
            self.c_qr = sp.sparse.linalg.spsolve_triangular(
                self.pyspqr_r.T, self.pyspqr_e @ getattr(
                    self, self.used_c), lower=True)
            # self.c_qr = self.triangular_solve_transpose.solve(
            #     getattr(self, self.used_c))
            # breakpoint()

        # shift in the linspace projector
        # self.e = self.nullspace @ self.nullspace.T @ (self.qr_matrix @ self.c_qr - getattr(self, self.used_b)) - self.qr_matrix @ self.c_qr
        self.e = -self.nullspace @ self.nullspace.T @ getattr(self, self.used_b) - self.qr_matrix @ self.c_qr

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

    def dr_step(self, z):
        """DR step."""
        y = self.cone_project(z)
        return self.linspace_project(2 * y - z) - y

    def iterate(self):
        """Simple Douglas Rachford iteration."""
        self.y[:] = self.cone_project(self.z)
        step = self.linspace_project(2 * self.y - self.z) - self.y
        # print(np.linalg.norm(step))
        self.z[:] = self.z + step
        # self.y[:] = self.cone_project(self.z)

        # self.allsteps.append(np.copy(step))

        #if len(self.solution_qualities) > 50000:
#         if  len(self.solution_qualities) > 10000 and (
#             self.solution_qualities[-1] > 1e-6) and (
#             (self.solution_qualities[-1] - self.solution_qualities[-2]) > -1e-10):
#             import matplotlib.pyplot as plt

# myz = self.z - step
# curstep = self.dr_step(myz)
# nextstep = self.dr_step(myz + curstep)
# nextnextstep = self.dr_step(myz + curstep + nextstep)
# plt.plot(curstep)
# plt.plot(nextstep)
# plt.plot(nextnextstep)
# plt.show()


# J = -np.eye(self.m)
# dz = self.dr_step(self.z)
# df = self.dr_step(self.z + dz) - self.dr_step(self.z)
# num = dz - J @ df
# den = dz @ J @ df
# J_plus = J + np.outer(num, dz @ J) / den

#             breakpoint()

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        self.y[:] = self.cone_project(self.z)
        self.s[:] = self.y - self.z
        x_qr = self.qr_matrix.T @ (getattr(self, self.used_b) - self.s)
        # breakpoint()
        if self.use_numpy:
            self.x[:] = sp.linalg.solve_triangular(self.triangular, x_qr, lower=False)
        else:
            # self.x[:] = self.triangular_solve.solve(x_qr)
            self.x[:] = self.pyspqr_e.T @ sp.sparse.linalg.spsolve_triangular(
                self.pyspqr_r, x_qr, lower=False)
            # breakpoint()
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

class BaseBroydenCQR(NewCQR):
    """Add logic to save dz's and dstep's."""

    memory = 10
    max_iterations = 100000
    myverbose = False
    sample_period = 1

    def prepare_loop(self):
        """Create storage arrays."""
        super().prepare_loop()
        self.dzs = np.empty((self.memory, self.m), dtype=float)
        self.dsteps = np.empty((self.memory, self.m), dtype=float)
        self.old_z = np.empty(self.m, dtype=float)
        self.old_step = np.empty(self.m, dtype=float)

    def iterate(self):
        """Simple Douglas Rachford iteration with Broyden update to override.
        """
        if self.memory == 0:
            step = self.dr_step(self.z)
            if self.myverbose:
                print(np.linalg.norm(step))
            self.z[:] += step
            return

        if len(self.solution_qualities) % self.sample_period == 0:
            self.dzs[
                len(self.solution_qualities) % self.memory] = self.z - self.old_z
            self.old_z[:] = self.z

        self.y[:] = self.cone_project(self.z)
        step = self.linspace_project(2 * self.y - self.z) - self.y
        if self.myverbose:
            print(np.linalg.norm(step))

        if len(self.solution_qualities) % self.sample_period == 0:
            self.dsteps[
                len(self.solution_qualities) % self.memory] = step - self.old_step
            self.old_step[:] = step

        if len(self.solution_qualities) > self.memory * self.sample_period + 2: # + 1 should suffice
            newstep = self.compute_broyden_step(step)
            # if len(self.solution_qualities) > 50000:
            #     breakpoint()
            self.z[:] = self.z - newstep
        else:
            self.z[:] = self.z + step

    def compute_broyden_step(self, step):
        """Base method to compute a Broyden-style approximate Newton step."""
        return -step

class QRBroydenCQR(BaseBroydenCQR):
    """Test with QR of the diffs."""
    max_iterations = 10000
    memory = 5
    def compute_broyden_step(self, step):
        """With QR."""

        mystep = np.copy(step)
        result = np.zeros_like(step)
        # breakpoint()

        # H @ self.dsteps.T = self.dzs.T
        q, r = np.linalg.qr(self.dsteps.T)
        # H @ (q @ r) = self.dzs.T
        # H @ q = self.dzs.T @ (r^-1)

        # assert np.allclose(
        #     sp.linalg.solve_triangular(r.T, self.dzs, lower=True).T,
        #     self.dzs.T @ np.linalg.inv(r))

        # rhs = sp.linalg.solve_triangular(
        #     r.T, self.dzs, lower=True).T
        # rhs = self.dzs.T @ np.linalg.inv(r)
        components = (q.T @ mystep)

        # remove components
        mystep -= q @ components
        # result += rhs @ components
        # result += self.dzs.T @ (np.linalg.inv(r) @ components)
        # breakpoint()
        result += self.dzs.T @ sp.linalg.solve_triangular(
           r, components, lower=False)

        # add rest
        result -= mystep

        return result

class QR3BroydenCQR(QRBroydenCQR):
    """To run long test, memory 3."""
    max_iterations = 100000
    memory = 3

class QRType1BroydenCQR(BaseBroydenCQR):
    """Test with QR of the diffs."""
    max_iterations = 10000
    memory = 5

    def compute_broyden_step(self, step):
        """With QR."""

        q, r = np.linalg.qr(self.dzs.T)
        # rhs = sp.linalg.solve_triangular(
        #     r.T, self.dsteps, lower=True).T
        # rhs = self.dsteps.T @ np.linalg.inv(r)
        # J = -np.eye(self.m) + (rhs + q) @ q.T
        # result = np.linalg.solve(J, step)
        # Jinv = -np.eye(self.m) - (rhs + q) @ np.linalg.inv(np.eye(self.memory) - q.T @ (rhs + q)) @ q.T
        # result = Jinv @ step
        # result = -step - (rhs + q) @ np.linalg.solve(
        #     np.eye(self.memory) - q.T @ (rhs + q),
        #     (q.T @ step))
        # result = -step + (rhs + q) @ np.linalg.solve(q.T @ rhs, q.T @ step)
        # result = -step + (rhs + q) @ (np.linalg.inv(q.T @ rhs) @ (q.T @ step))
        # result = -step + (rhs + q) @ (np.linalg.inv(q.T @ self.dsteps.T @ np.linalg.inv(r)) @ (q.T @ step))
        # result = -step + (rhs + q) @ r @ ((np.linalg.inv(q.T @ self.dsteps.T)) @ (q.T @ step))
        # result = -step + (self.dsteps.T @ np.linalg.inv(r) + q) @ r @ np.linalg.solve(q.T @ self.dsteps.T, q.T @ step)
        # result = -step + (self.dsteps.T  + q @ r) @ np.linalg.solve(q.T @ self.dsteps.T, q.T @ step)
        result = -step + (self.dsteps.T  + self.dzs.T) @ np.linalg.solve(q.T @ self.dsteps.T, q.T @ step)
        return result


class QRNormBroydenCQR(BaseBroydenCQR):
    """Test with QR of the diffs and diagonal normalization."""
    max_iterations = 100000
    memory = 3
    def compute_broyden_step(self, step):
        """With QR."""
        # print(np.linalg.norm(step))

        # diagonal equilibration
        dsteps_norm = np.linalg.norm(self.dsteps, axis=0)
        # dsteps_norm[:] = 1.
        dsteps_norm[dsteps_norm == 0.] = 1.

        dzs_norm = np.linalg.norm(self.dzs, axis=0)
        # dzs_norm[:] = 1.
        dzs_norm[dzs_norm == 0.] = 1.

        mystep = np.copy(step)
        result = np.zeros_like(step)

        # H @ diag(dsn) @ diag(dsn^-1) @ self.dsteps.T = diag(dzn) @ diag(dzn^-1) @ self.dzs.T
        q, r = np.linalg.qr((self.dsteps / dsteps_norm).T)
        # q @ r = diag(dsn^-1) @ self.dsteps.T

        rhs = sp.linalg.solve_triangular(
            r.T, (self.dzs / dzs_norm),
            lower=True).T
        # rhs = diag(dzn^-1) @ self.dzs.T @ np.linalg.inv(r)

        # so
        # H @ diag(dsn) @ q = diag(dzn) @ rhs
        # (diag(dzn^-1) H @ diag(dsn)) @ q = rhs
        # Hscal = (diag(dzn^-1) @ H @ diag(dsn))
        # Hscal @ q = rhs
        # we have
        # H0 = -I
        # H0scal = (diag(dzn^-1) @ -I @ diag(dsn)) = -diag(dsn/dzn)
        # so
        # Hscal = H0scal + rhs @ q.T - H0scal @ q @ q.T
        CVXPY = False
        if CVXPY:
            import cvxpy as cp
            H = cp.Variable((self.m, self.m))
            H0 = -np.eye(self.m)
            Hscal = np.diag(dzs_norm**-1) @ H @ np.diag(dsteps_norm)
            H0scal = np.diag(dzs_norm**-1) @ H0 @ np.diag(dsteps_norm)
            objective = cp.Minimize(cp.sum_squares(Hscal - H0scal))
            constraints = [Hscal @ q == rhs]
            cp.Problem(objective, constraints).solve()
            assert np.allclose(Hscal.value @ q, rhs)
            result =  H.value @ mystep
        else:
            # H0 = -np.eye(self.m)
            # H0scal = np.diag(dzs_norm**-1) @ H0 @ np.diag(dsteps_norm)
            # Hscal = H0scal + rhs @ q.T - H0scal @ q @ q.T
            # H = np.diag(dzs_norm) @ Hscal @ np.diag(dsteps_norm**-1)
            # H = np.diag(dzs_norm) @ (H0scal + rhs @ q.T - H0scal @ q @ q.T) @ np.diag(dsteps_norm**-1)
            # H = H0 + np.diag(dzs_norm) @ (rhs @ q.T - H0scal @ q @ q.T) @ np.diag(dsteps_norm**-1)

            # result = H @ mystep

            # let's rebuild the result
            result -= mystep # here we use H0
            mystep_norm = mystep / dsteps_norm
            components = q.T @ mystep_norm
            result += (rhs @ components) * dzs_norm
            result += (q @ components) * dsteps_norm # here we use H0

        # print(result)
        return result
        # breakpoint()

        # we used to have:
        # H = H0 (I - np.outer(q, q)) + np.outer(rhs, q)

        components = (q.T @ (mystep))

        # remove components
        mystep -= q @ components
        result += rhs @ components

        # add rest
        result -= mystep

        return result

class SparseBroydenCQR(BaseBroydenCQR):
    """Test sparse 1-memory."""
    max_iterations = 100000
    memory = 1
    # def compute_broyden_step(self, step):
    #     """1-Memory sparse update."""

    #     J0 = sp.sparse.linalg.LinearOperator(
    #         shape=(self.m, self.m),
    #         matvec = lambda x: -x
    #     )

    #     ds = self.dsteps[0,:]
    #     ds_norm = np.linalg.norm(ds)
    #     ds_normed = ds / ds_norm
    #     dz = self.dzs[0,:]
    #     dz_snormed = dz / ds_norm

    #     def matvec(x, ds_normed, dz_snormed, J_minus1):
    #         ds_component = x @ ds_normed
    #         return J_minus1 @ (
    #             x - ds_normed * ds_component) + dz_snormed * ds_component

    #     J1 = sp.sparse.linalg.LinearOperator(
    #         shape=(self.m, self.m),
    #         matvec = lambda x: matvec(x,  ds_normed, dz_snormed, J0)
    #     )
    #     # breakpoint()
    #     assert np.allclose(J1 @ self.dsteps[0], self.dzs[0])
    #     return J1 @ step

    def compute_broyden_step(self, step):
        """1-Memory sparse update."""

        mystep = np.copy(step)
        result = np.zeros_like(step)

        # correction by current index
        ds = self.dsteps[0, :]
        ds_norm = np.linalg.norm(ds)
        ds_normed = ds / ds_norm
        dz = self.dzs[0, :]
        dz_snormed = dz / ds_norm

        ds_component = mystep @ ds_normed
        mystep -= ds_normed * ds_component
        result +=  dz_snormed * ds_component

        # final correction
        result -= mystep

        # # cur step len
        # cur_step_len = np.linalg.norm(step)
        # # next step len
        # next_step_len = np.linalg.norm(self.dr_step(self.z - result))
        # if next_step_len > 9 * cur_step_len:
        #     breakpoint()

        return result


class SparseSoftBroydenCQR(BaseBroydenCQR):
    """Test 1-memory dividing by 2 the update."""
    max_iterations = 100000
    memory = 1

    def compute_broyden_step(self, step):
        """1-Memory sparse update."""

        mystep = np.copy(step)
        result = np.zeros_like(step)

        # correction by current index
        ds = self.dsteps[0, :]
        ds_norm = np.linalg.norm(ds)
        ds_normed = ds / ds_norm
        dz = self.dzs[0, :]
        dz_norm = np.linalg.norm(dz)
        # print('norm(dz)/norm(ds)', dz_norm/ds_norm)
        # dz_snormed = dz / ds_norm
        dz_normed = dz / dz_norm
        # this is how much we accelerate along rank1 Jinverse
        scale_factor = dz_norm / ds_norm

        # this is how much we reduce the correction
        reducer = self.scale_reducer(scale_factor)

        ds_component = (mystep @ ds_normed)
        # remove component along direction
        mystep -= ds_normed * (ds_component / reducer)
        # add rank1 J1 inverse piece
        result +=  (ds_component * scale_factor / reducer) * dz_normed

        # final correction
        result -= mystep

        return result

    def scale_reducer(self, scale_factor):
        """Test scale reducer."""
        return 2.

class SparseSoftSqrtBroydenCQR(SparseSoftBroydenCQR):
    """Test 1-memory with sqrt scaling update."""

    def scale_reducer(self, scale_factor):
        """Test scale reducer."""
        return np.sqrt(scale_factor)

class SparseSoftInvCorrBroydenCQR(SparseSoftBroydenCQR):
    """Test 1-memory with inverse correction scaling update."""

    correction_scale = 100.

    def scale_reducer(self, scale_factor):
        """Test scale reducer."""
        return 1./(1./scale_factor + 1./self.correction_scale)


class Sparse2BroydenCQR(BaseBroydenCQR):
    """Test sparse 2-memory."""
    max_iterations = 100_000
    memory = 2
    def compute_broyden_step(self, step):
        """2-Memory sparse update."""

        mystep = np.copy(step)
        result = np.zeros_like(step)

        # this should be correct
        current_index = len(self.solution_qualities) % self.memory
        previous_index = 1 - current_index

        # correction by current index
        ds = self.dsteps[current_index, :]
        ds_norm = np.linalg.norm(ds)
        ds_normed = ds / ds_norm
        dz = self.dzs[current_index, :]
        dz_snormed = dz / ds_norm

        ds_component = mystep @ ds_normed
        mystep -= ds_normed * ds_component
        result +=  dz_snormed * ds_component

        # correction by previous index
        ds = self.dsteps[previous_index, :]
        ds_norm = np.linalg.norm(ds)
        ds_normed = ds / ds_norm
        dz = self.dzs[previous_index, :]
        dz_snormed = dz / ds_norm

        ds_component = mystep @ ds_normed
        mystep -= ds_normed * ds_component
        result +=  dz_snormed * ds_component

        # final correction
        result -= mystep

        return result

class Sparse3BroydenCQR(BaseBroydenCQR):
    """Test sparse 2-memory."""
    max_iterations = 10_000
    memory = 3
    def compute_broyden_step(self, step):
        """2-Memory sparse update."""

        mystep = np.copy(step)
        result = np.zeros_like(step)

        # this should be correct
        current_index = len(self.solution_qualities) % self.memory
        previous_index = (len(self.solution_qualities)-1) % self.memory
        previous_previous_index = (len(self.solution_qualities)-2) % self.memory

        # correction by current index
        ds = self.dsteps[current_index, :]
        ds_norm = np.linalg.norm(ds)
        ds_normed = ds / ds_norm
        dz = self.dzs[current_index, :]
        dz_snormed = dz / ds_norm

        ds_component = mystep @ ds_normed
        mystep -= ds_normed * ds_component
        result +=  dz_snormed * ds_component

        # correction by previous index
        ds = self.dsteps[previous_index, :]
        ds_norm = np.linalg.norm(ds)
        ds_normed = ds / ds_norm
        dz = self.dzs[previous_index, :]
        dz_snormed = dz / ds_norm

        ds_component = mystep @ ds_normed
        mystep -= ds_normed * ds_component
        result +=  dz_snormed * ds_component

        # correction by previous previous index
        ds = self.dsteps[previous_previous_index, :]
        ds_norm = np.linalg.norm(ds)
        ds_normed = ds / ds_norm
        dz = self.dzs[previous_previous_index, :]
        dz_snormed = dz / ds_norm

        ds_component = mystep @ ds_normed
        mystep -= ds_normed * ds_component
        result +=  dz_snormed * ds_component

        # final correction
        result -= mystep

        cur_obj = np.linalg.norm(step)
        # backtrack
        for i in range(10):
            next_obj = np.linalg.norm(self.dr_step(self.z - result))
            if next_obj < cur_obj:
                # print(f'IMPROVED WITH {i} BACKTRACKS')
                break
            result /= 2.
        else:
            # print('REACHED MAX NUM BACKTRACKS, RETURNING STEP')
            return -step

        return result

class SparseNBacktrackedBroydenCQR(BaseBroydenCQR):
    """Full memory BrCQR, with simple backtrack."""
    max_iterations = 10_000
    memory = 10
    def compute_broyden_step(self, step):
        """2-Memory sparse update."""

        mystep = np.copy(step)
        result = np.zeros_like(step)
        # diagnostic
        components = np.zeros(self.memory)
        accelerations = np.zeros(self.memory)

        # this should be correct
        for back_index in range(self.memory):
            current_index = (len(
                self.solution_qualities) - back_index) % self.memory

            # correction by current index
            ds = self.dsteps[current_index, :]
            ds_norm = np.linalg.norm(ds)
            ds_normed = ds / ds_norm
            dz = self.dzs[current_index, :]
            dz_snormed = dz / ds_norm

            ds_component = mystep @ ds_normed
            components[back_index] = ds_component
            accelerations[back_index] = np.linalg.norm(dz_snormed)
            mystep -= ds_normed * ds_component
            result +=  dz_snormed * ds_component

        # final correction
        result -= mystep

        cur_obj = np.linalg.norm(step)
        # backtrack
        for i in range(10):
            next_obj = np.linalg.norm(self.dr_step(self.z - result))
            if next_obj < cur_obj:
                # print(f'IMPROVED WITH {i} BACKTRACKS')
                if (len(self.solution_qualities) > 2000) and i == 0:
                    print('0 BACKTRACKS!')
                    print('ITER')
                    print(len(self.solution_qualities))
                    print('COMPONENTS * ACCELERATIONS / norm(step)')
                    print(
                        (components * accelerations) / np.linalg.norm(step)
                    )
                break
            result /= 2.
        else:
            if len(self.solution_qualities) > 2000:
                print('REACHED MAX NUM BACKTRACKS, RETURNING STEP')
                print('ITER')
                print(len(self.solution_qualities))
                # print('COMPONENTS/norm(step)')
                # print(components/np.linalg.norm(step))
                # print('ACCELERATIONS', accelerations)
                print('COMPONENTS * ACCELERATIONS / norm(step)')
                print(
                    (components * accelerations) / np.linalg.norm(step)
                )
                breakpoint()
            return -step

        return result

class SparseNTestBroydenCQR(BaseBroydenCQR):
    """Full memory BrCQR, testing normalization."""
    max_iterations = 100_000
    memory = 20
    acceleration_cap = 20
    def compute_broyden_step(self, step):
        """N-Memory sparse update."""

        mystep = np.copy(step)
        result = np.zeros_like(step)
        # step_norm = np.linalg.norm(step)
        # diagnostic
        # components = np.zeros(self.memory)
        # accelerations = np.zeros(self.memory)

        # this should be correct
        for back_index in range(self.memory):
            current_index = (len(
                self.solution_qualities) - back_index) % self.memory

            # correction by current index
            ds = self.dsteps[current_index, :]
            ds_norm = np.linalg.norm(ds)
            ds_normed = ds / ds_norm
            dz = self.dzs[current_index, :]
            dz_norm = np.linalg.norm(dz)
            # dz_normed = dz / dz_norm
            dz_snormed = dz / ds_norm
            acceleration = dz_norm / ds_norm

            # we cap the acceleration
            if acceleration > self.acceleration_cap:
                reduction_factor = acceleration / self.acceleration_cap
                # breakpoint()
            else:
                reduction_factor = 1.

            ds_component_reduced = (mystep @ ds_normed) / reduction_factor
            # components[back_index] = ds_component
            # accelerations[back_index] = np.linalg.norm(dz_snormed)
            mystep -= ds_normed * ds_component_reduced
            result +=  (dz_snormed * ds_component_reduced)
        # final correction
        result -= mystep
        return result

class SparseNTestAdaCapBroydenCQR(BaseBroydenCQR):
    """Test same acceleration cap but with adaptive scheme.
    
    (case 1) Cap not hit; do nothing.
    (case 2) Cap hit and len of next step greater than curret; decrease cap;
        accept this step anyways for now (so we don't increase overall cost).
    (case 3) Cap hit and len of n.s. smaller; accept update; increase cap.
    """
    max_iterations = 100_000
    memory = 20

    # initial value
    acceleration_cap = 5
    cap_decrease_factor = 0.9
    cap_increase_factor = 1.005
    cap_floor = 1.
    cap_ceil = 50.

    def compute_broyden_step(self, step):
        """N-Memory sparse update."""

        cap_hit = False

        mystep = np.copy(step)
        current_step_len = np.linalg.norm(step)

        result = np.zeros_like(step)
        # step_norm = np.linalg.norm(step)
        # diagnostic
        # components = np.zeros(self.memory)
        # accelerations = np.zeros(self.memory)

        # this should be correct
        for back_index in range(self.memory):
            current_index = (len(
                self.solution_qualities) - back_index) % self.memory

            # correction by current index
            ds = self.dsteps[current_index, :]
            ds_norm = np.linalg.norm(ds)
            ds_normed = ds / ds_norm
            dz = self.dzs[current_index, :]
            dz_norm = np.linalg.norm(dz)
            # dz_normed = dz / dz_norm
            dz_snormed = dz / ds_norm
            acceleration = dz_norm / ds_norm

            # we cap the acceleration
            if acceleration > self.acceleration_cap:
                reduction_factor = acceleration / self.acceleration_cap
                cap_hit = True
            else:
                reduction_factor = 1.

            ds_component_reduced = (mystep @ ds_normed) / reduction_factor
            # components[back_index] = ds_component
            # accelerations[back_index] = np.linalg.norm(dz_snormed)
            mystep -= ds_normed * ds_component_reduced
            result +=  (dz_snormed * ds_component_reduced)
        # final correction
        result -= mystep

        # logic for update
        if cap_hit:
            # with refactoring we won't need to calculate this twice
            next_step_len = np.linalg.norm(self.dr_step(self.z - result))
            it = len(self.solution_qualities)
            if next_step_len < current_step_len:
                # print(f'ITER {it} CURRENT ACCEL {self.acceleration_cap}, INCREASING')
                self.acceleration_cap *= self.cap_increase_factor
                self.acceleration_cap = np.minimum(self.cap_ceil, self.acceleration_cap)
            else:
                # print(f'ITER {it} CURRENT ACCEL {self.acceleration_cap}, DECREASING')
                self.acceleration_cap *= self.cap_decrease_factor
                self.acceleration_cap = np.maximum(self.cap_floor, self.acceleration_cap)
        return result


class SparseNAltAccelCapBroydenCQR(BaseBroydenCQR):
    """Full memory BrCQR, testing normalization."""
    max_iterations = 100_000
    memory = 20
    alt_acceleration_cap = 20
    def compute_broyden_step(self, step):
        """N-Memory sparse update."""

        mystep = np.copy(step)
        result = np.zeros_like(step)
        # step_norm = np.linalg.norm(step)
        # diagnostic
        # components = np.zeros(self.memory)
        # accelerations = np.zeros(self.memory)

        # this should be correct
        for back_index in range(self.memory):
            current_index = (len(
                self.solution_qualities) - back_index) % self.memory

            # correction by current index
            ds = self.dsteps[current_index, :]
            ds_norm = np.linalg.norm(ds)
            ds_normed = ds / ds_norm
            dz = self.dzs[current_index, :]
            dz_norm = np.linalg.norm(dz)
            # dz_normed = dz / dz_norm
            dz_snormed = dz / ds_norm
            acceleration = dz_norm / ds_norm

            nrm_mystep = np.linalg.norm(mystep)
            exposure_accel = acceleration * ((mystep @ ds_normed) / nrm_mystep)
            exposure_accel_capped = np.clip(
                exposure_accel,
                min=-self.alt_acceleration_cap, max=self.alt_acceleration_cap)
            if exposure_accel != exposure_accel_capped:
                print(exposure_accel, exposure_accel_capped)
            ds_component_reduced = (exposure_accel_capped * nrm_mystep) / acceleration

            # ds_component_reduced = (mystep @ ds_normed) / reduction_factor
            # components[back_index] = ds_component
            # accelerations[back_index] = np.linalg.norm(dz_snormed)
            mystep -= ds_normed * ds_component_reduced
            result +=  (dz_snormed * ds_component_reduced)
        # final correction
        result -= mystep
        return result

class SparseTestTestBroydenCQR(SparseNTestBroydenCQR):
    """Full memory BrCQR, testing normalization."""
    max_iterations = 100_000
    memory = 10
    acceleration_cap = 100

class DenseBroydenCQR(BaseBroydenCQR):
    """Add logic to save dz's and dstep's."""
    max_iterations = 100
    memory = 1
    def compute_broyden_step(self, step):
        """1-Memory dense update."""
        J0 = -np.eye(self.m)
        ds = self.dsteps[0, :]
        dz = self.dzs[0, :]
        corr_1 = np.outer(dz/np.linalg.norm(ds), ds/np.linalg.norm(ds))
        corr_2 = np.outer(J0 @ (ds/np.linalg.norm(ds)), ds/np.linalg.norm(ds))
        J1 = J0 + corr_1 - corr_2
        assert np.allclose(J1 @ self.dsteps.T, self.dzs.T)
        return J1 @ step


class ToyBroydenCQR(BaseBroydenCQR):
    """Temporary."""

    def compute_broyden_step(self, step):
        """Temporary."""
        import cvxpy as cp
        Jacobian = cp.Variable((self.m, self.m))
        objective = cp.Minimize(cp.sum_squares(Jacobian + np.eye(self.m)))
        # constraints = [Jacobian @ self.dzs.T == self.dsteps.T]
        constraints = [Jacobian @ self.dsteps.T == self.dzs.T]
        cp.Problem(objective, constraints).solve(verbose=False)
        # newstep = np.linalg.solve(Jacobian.value, step)
        newstep = Jacobian.value @ step
        return newstep

class LevMarNewCQR(NewCQR):
    """Using Levemberg Marquardt."""

    lsqr_iters = 5
    max_iterations = 100000//(2 * lsqr_iters + 2)
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

class QRLevMarBroydenCQR(BaseBroydenCQR, LevMarNewCQR):
    """Test with QR of the diffs."""
    lsqr_iters = 2
    max_iterations = 100000 // (2 * lsqr_iters + 2)
    memory = 1
    def compute_broyden_step(self, step):
        """With QR."""
        # print(np.linalg.norm(step))
        mystep = np.copy(step)
        result = np.zeros_like(step)
        # breakpoint()

        # H @ self.dsteps.T = self.dzs.T
        q, r = np.linalg.qr(self.dsteps.T)
        # H @ (q @ r) = self.dzs.T
        # H @ q = self.dzs.T @ (r^-1)

        # assert np.allclose(
        #     sp.linalg.solve_triangular(r.T, self.dzs, lower=True).T,
        #     self.dzs.T @ np.linalg.inv(r))

        # rhs = sp.linalg.solve_triangular(
        #    r.T, self.dzs, lower=True).T
        # rhs = self.dzs.T @ np.linalg.inv(r)
        components = (q.T @ mystep) / 1.

        # remove components
        mystep -= q @ components
        # result += rhs @ components
        result += self.dzs.T @ sp.linalg.solve_triangular(
           r, components, lower=False)
        # result += self.dzs.T @ np.linalg.inv(r) @ components
        # breakpoint()
        # result += self.dzs.T @ sp.linalg.solve_triangular(
        #     r, components, lower=False)

        # add rest
        # result -= mystep
        # return result

        result_levmar = sp.sparse.linalg.lsqr(
            sp.sparse.linalg.LinearOperator(
                shape=(self.m, self.m),
                matvec=lambda dz: self.multiply_jacobian_dstep(self.z, dz),
                rmatvec=lambda dr: self.multiply_jacobian_dstep_transpose(
                    self.z, dr)), -mystep,
                    x0=mystep,
                    damp=0., # might make sense to change this?
                    atol=0., btol=0., # might make sense to change this
                    iter_lim=self.lsqr_iters)
        # breakpoint()
        # add final part
        result -= result_levmar[0]

        return result

class SparseNTestLevMarBroydenCQR(BaseBroydenCQR, LevMarNewCQR):
    """Full memory BrCQR, testing normalization."""
    max_iterations = 100_000
    memory = 10
    acceleration_cap = 20
    lsqr_iters = 1
    max_iterations = 100000 // (2 * lsqr_iters + 2)
    def compute_broyden_step(self, step):
        """N-Memory sparse update."""

        mystep = np.copy(step)
        result = np.zeros_like(step)
        # step_norm = np.linalg.norm(step)
        # diagnostic
        # components = np.zeros(self.memory)
        # accelerations = np.zeros(self.memory)

        # this should be correct
        for back_index in range(self.memory):
            current_index = (len(
                self.solution_qualities) - back_index) % self.memory

            # correction by current index
            ds = self.dsteps[current_index, :]
            ds_norm = np.linalg.norm(ds)
            ds_normed = ds / ds_norm
            dz = self.dzs[current_index, :]
            dz_norm = np.linalg.norm(dz)
            # dz_normed = dz / dz_norm
            dz_snormed = dz / ds_norm
            acceleration = dz_norm / ds_norm

            # we cap the acceleration
            if acceleration > self.acceleration_cap:
                reduction_factor = acceleration / self.acceleration_cap
                # breakpoint()
            else:
                reduction_factor = 1.

            ds_component_reduced = (mystep @ ds_normed) / reduction_factor
            # components[back_index] = ds_component
            # accelerations[back_index] = np.linalg.norm(dz_snormed)
            mystep -= ds_normed * ds_component_reduced
            result +=  (dz_snormed * ds_component_reduced)
        # final correction
        result_levmar = sp.sparse.linalg.lsqr(
            sp.sparse.linalg.LinearOperator(
                shape=(self.m, self.m),
                matvec=lambda dz: self.multiply_jacobian_dstep(self.z, dz),
                rmatvec=lambda dr: self.multiply_jacobian_dstep_transpose(
                    self.z, dr)), -mystep,
                    x0=mystep,
                    damp=0., # might make sense to change this?
                    atol=0., btol=0., # might make sense to change this
                    iter_lim=self.lsqr_iters)
        # breakpoint()
        # add final part
        result -= result_levmar[0]

        return result

class LevMarCGNewCQR(LevMarNewCQR):
    """Using Levemberg Marquardt, with explicit CG formulation."""

    cg_iters = 100
    max_iterations = 100000
    matmul_count = 1 # used below

    # def prepare_loop(self):
    #     """Test preconditioner."""
    #     super().prepare_loop()
    #     self.preconditioner = np.linalg.pinv(
    #         self.nullspace @ (self.nullspace.T))
    #     # breakpoint()

    def cg_multiply(self, z, input):
        """CG matrix multiplication."""
        return self.multiply_jacobian_dstep_transpose(
            z, self.multiply_jacobian_dstep(z, input))

    def iterate(self):
        """Do one iteration."""

        self.y[:] = self.cone_project(self.z)
        step = self.linspace_project(2 * self.y - self.z) - self.y
        # print(np.linalg.norm(step))

        self.matmul_count = 2

        def _callback(_):
            self.matmul_count += 1

        cg_matrix = sp.sparse.linalg.LinearOperator(
            shape=(self.m, self.m),
            matvec=lambda input: self.cg_multiply(self.z, input))
        rhs = -self.multiply_jacobian_dstep_transpose(self.z, step)

        result = sp.sparse.linalg.cg(
            cg_matrix, rhs, x0=step,
            rtol=min(0.5, np.sqrt(np.linalg.norm(rhs))),
            maxiter=min(self.cg_iters, len(self.solution_qualities)),
            # M = self.preconditioner,
            callback=_callback)
        # breakpoint()
        # print(result[-1])
        self.z[:] = self.z + result[0]

        if len(self.solution_qualities) > 20000:
            import matplotlib.pyplot as plt

            myz = self.z - result[0]
            for i in range(10):
                curstep = self.dr_step(myz)
                plt.plot(curstep, label=f'step {i}')
                myz += curstep
            plt.legend()
            plt.show()

            cg_matrix = self._densify_square(cg_matrix)
            plt.imshow(cg_matrix)
            plt.colorbar()
            plt.title("CG MATRIX")
            plt.figure()
            plt.plot(rhs)
            plt.title("RHS")
            plt.figure()
            plt.plot(step)
            plt.title("STEP")
            plt.show()
            breakpoint()

    @staticmethod
    def _densify_square(linear_operator):
        """Create Numpy 2-d array from a sparse LinearOperator."""
        assert linear_operator.shape[0] == linear_operator.shape[1]
        result = np.eye(linear_operator.shape[0], dtype=float)
        for i in range(len(result)):
            result[:, i] = linear_operator.matvec(result[:, i])
        return result

    def callback_iterate(self):
        """You can probably re-use this with custom loops.

        :raises StopIteration:
        """
        self.obtain_x_and_y()
        # extend
        if self.matmul_count > 1:
            self.solution_qualities += [self.solution_qualities[-1]] * (
                self.matmul_count - 1)
        if len(self.solution_qualities) > self.max_iterations:
            raise StopIteration
        self.solution_qualities.append(self.check_solution_quality())
        if self.solution_qualities[-1] < self.epsilon_convergence:
            raise StopIteration

    # S(z) = PiLin(2 * PiCon(z) - z) - PiCon(z)
    # DS = DLin @ (2 * DCon - I) - DCon
    #
    # Useful identities:
    # DLin = DLin.T
    # DCon = Dcon.T # I guess also extends to non-symmetric cones
    # DLin @ Dlin = DLin # not true for the cone one apparently
    # Dlin @ PiLin(z) = PiLin(z)
    # DCon @ PiCon(z) = PiCon(z)
    #
    # We need:
    # DS.T @ DS
    # DS.T @ S(z)
    #
    # It appears we don't save anything with this
    #
    # DS.T @ DS = ((2 * DCon - I) @ DLin - DCon) @ (DLin @ (2 * DCon - I) - DCon)
    # = (2 * DCon @ DLin - Dlin - DCon) @ (DLin @ (2 * DCon) - Dlin - DCon)
    # = 4 * DCon @ DLin @ DCon - 2 * Dcon @ Dlin - 2 * Dcon @ Dlin @ Dcon
    #   - 2 * Dlin  @ Dcon + Dlin + Dlin @ Dcon
    #   - 2 * Dcon @ Dlin @ Dcon + Dcon @ Dlin + Dcon @ Dcon
    # = - 2 * Dcon @ Dlin - 2 * Dlin @ Dcon + Dlin + Dlin @ Dcon + Dcon @ Dlin + Dcon @ Dcon
    # = - Dcon @ Dlin - Dlin @ Dcon + Dlin + Dcon @ Dcon
    # = (-DCon + I) @ (DLin) @ (-DCon + I) + (DCon) @ (DLin) @ (DCon) + Dcon @ Dcon
    # = (-DCon + I) @ (DLin + I) @ (-DCon + I) + (DCon) @ (DLin) @ (DCon) - I + 2 * DCon
    #
    # DS.T @ S(z) = ((2 * DCon - I) @ DLin - DCon) @ (PiLin(2 * PiCon(z) - z) - PiCon(z))
    # = (2 * DCon @ DLin - Dlin - DCon) @ (PiLin(2 * PiCon(z) - z) - PiCon(z))
    # = (2 * DCon @ PiLin(2 * PiCon(z) - z) - PiLin(2 * PiCon(z) - z) - DCon @ PiLin(2 * PiCon(z) - z)
    #   - 2 * DCon @ DLin @ PiCon(z) + DLin @ PiCon(z) + PiCon(z)


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
    ruiz_rounds = 100
    ruiz_norm = np.inf

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
            if self.ruiz_norm == np.inf:
                return np.max(np.abs(concatenated), axis=0)
            if self.ruiz_norm == 2:
                return np.linalg.norm(concatenated, axis=0)

        def norm_rows(concatenated):
            if self.ruiz_norm == np.inf:
                return np.max(np.abs(concatenated), axis=1)
            if self.ruiz_norm == 2:
                return np.linalg.norm(concatenated, axis=1)

        m, n = matrix.shape

        d_and_rho = np.ones(m+1)
        e_and_sigma = np.ones(n+1)

        for i in range(self.ruiz_rounds):

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
            print(r1, r2)
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


class AlternativeEquilibration(NewCQR):
    """Idea of alternative eq scheme."""
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

        m = sp.sparse.coo_array(work_matrix)
        logvals = np.log(np.abs(m.data))
        import cvxpy as cp
        x, y = m.coords
        ex = cp.Variable(m.shape[0])
        ey = cp.Variable(m.shape[1])
        equilibrated = logvals + ex[x] + ey[y]
        x_idx = np.arange(len(x))
        y_idx = np.arange(len(y))
        # cp.Problem(cp.Minimize(cp.max(equilibrated) - cp.min(equilibrated))).solve(solver='ECOS', verbose=True)
        objective = 0
        x_conds = []
        for my_x_idx in range(len(x)):
            used_idxs = x_idx[x == my_x_idx]
            if len(used_idxs) == 0:
                continue
            x_conds.append(cp.max(equilibrated[used_idxs]) - cp.min(equilibrated[used_idxs]))
        y_conds = []
        for my_y_idx in range(len(y)):
            used_idxs = y_idx[y == my_y_idx]
            if len(used_idxs) == 0:
                continue
            y_conds.append(cp.max(equilibrated[used_idxs]) - cp.min(equilibrated[used_idxs]))
        conds = cp.hstack(x_conds + y_conds)
        cp.Problem(cp.Minimize(cp.max(conds))).solve(solver='ECOS', verbose=True)
        # breakpoint()
        # cp.Problem(cp.Minimize(objective)).solve(solver='ECOS', verbose=True)
        # import matplotlib.pyplot as plt
        # plt.plot(ex.value)
        # plt.plot(ey.value)
        # plt.show()
        my_d = np.exp(ex.value)
        my_e = np.exp(ey.value)
        my_equilibration  = ((work_matrix * my_e).T * my_d).T
        # breakpoint()
        print("SORTED D", np.sort(my_d))
        print('SORTED E', np.sort(my_e))
        self.equil_e = my_e[:-1]
        self.equil_d = my_d[:-1]
        self.equil_sigma = my_e[-1]
        self.equil_rho = my_d[-1]

        self.eq_matrix = sp.sparse.csc_matrix(my_equilibration[:-1, :-1])
        self.eq_b = my_equilibration[:-1, -1]
        self.eq_c = my_equilibration[-1, :-1]

        super().prepare_loop()

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        super().obtain_x_and_y()

        self.x = (self.equil_e * self.x) / self.equil_sigma
        self.y = (self.equil_d * self.y) / self.equil_rho

class SparseNTestEquilibratedBroydenCQR(SparseNTestBroydenCQR, EquilibratedNewCQR):
    """Full memory BrCQR, testing acceleration cap, Ruiz of program data."""
    # best test so far 2025-09-27
    max_iterations = 100_000
    memory = 20
    acceleration_cap = 20
    ruiz_rounds = 2
    ruiz_norm = np.inf

class SparseNTestAltEquilibratedBroydenCQR(SparseNTestBroydenCQR, AlternativeEquilibration):
    """Full memory BrCQR, testing acceleration cap, Ruiz of program data."""
    max_iterations = 10_000
    memory = 20
    acceleration_cap = 20.
    myverbose = False
    # ruiz_rounds = 2
    # ruiz_norm = np.inf

class SparseNAltAccelCapEqBroydenCQR(SparseNAltAccelCapBroydenCQR, EquilibratedNewCQR):
    """Full memory BrCQR, testing acceleration cap, Ruiz of program data."""
    max_iterations = 100_000
    memory = 20
    alt_acceleration_cap = 5.
    myverbose = False
    ruiz_rounds = 2
    ruiz_norm = np.inf

class SparseNTestEquilibrated1BroydenCQR(SparseNTestEquilibratedBroydenCQR):
    """Alternative test, more ruiz."""
    max_iterations = 100_000
    memory = 20
    acceleration_cap = 20
    ruiz_rounds = 5
    ruiz_norm = np.inf

class SparseNTestEquilibrated2BroydenCQR(SparseNTestEquilibratedBroydenCQR):
    """Alternative test, less accel cap."""
    max_iterations = 100_000
    memory = 20
    acceleration_cap = 10
    ruiz_rounds = 2
    ruiz_norm = np.inf

class SparseNTestEquilibrated3BroydenCQR(SparseNTestEquilibratedBroydenCQR):
    """Alternative test, more accel cap."""
    max_iterations = 100_000
    memory = 20
    acceleration_cap = 50
    ruiz_rounds = 2
    ruiz_norm = np.inf

class SparseNTestEquilibrated4BroydenCQR(SparseNTestBroydenCQR, EquilibratedNewCQR):
    """Alternative test, more memory."""
    max_iterations = 100_000
    memory = 50
    acceleration_cap = 10
    ruiz_rounds = 2
    ruiz_norm = np.inf

class SparseNTestAdaCapEquilibratedBroydenCQR(SparseNTestAdaCapBroydenCQR, EquilibratedNewCQR):
    """With adaptive cap."""
    # best test so far, 2025-09-28
    max_iterations = 100_000
    memory = 20
    ruiz_rounds = 2
    ruiz_norm = np.inf

    # it has more parameters...
    acceleration_cap = 5 # initial value
    cap_decrease_factor = 0.9
    cap_increase_factor = 1.005
    cap_floor = 1.
    cap_ceil = 50.

class SparseNTestEquilibrated6BroydenCQR(SparseNTestAdaCapEquilibratedBroydenCQR):
    """Higher ceil."""
    # new best test so far, 2025-09-28
    cap_ceil = 100.

class SparseNTestEquilibrated5BroydenCQR(SparseNTestAdaCapBroydenCQR, EquilibratedNewCQR):
    """Alternative test, increasing sample period 2."""
    max_iterations = 100_000
    memory = 20
    ruiz_rounds = 2
    ruiz_norm = np.inf
    sample_period = 1

    def prepare_loop(self):
        super().prepare_loop()
        self.step_lens = []

    def compute_broyden_step(self, step):

        self.step_lens.append(np.linalg.norm(step))
        # breakpoint()
        if len(self.step_lens) > 2 and np.abs(np.diff(np.log(self.step_lens[-2:]))[0]) < 1e-6: # we're stuck
            if self.sample_period != 2:
                print('ITER', len(self.solution_qualities), 'SETTING PERIOD TO HIGH')
                self.sample_period = 2
                # import matplotlib.pyplot as plt
                # plt.plot(np.diff(np.log(self.step_lens))); plt.show()
                # plt.plot(self.dzs @ np.random.randn(self.m))
                # plt.plot(self.dsteps @ np.random.randn(self.m));
                # plt.show()
                # breakpoint()
        else:
            if self.sample_period != 1:
                print('ITER', len(self.solution_qualities), 'SETTING PERIOD TO 1')
            self.sample_period = 1
        return super().compute_broyden_step(step)

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


class EquilibratedLevMarNewCQR10Iter(EquilibratedNewCQR, LevMarNewCQR):
    """Equilibrated Lev Mar."""

    lsqr_iters = 10
    max_iterations = 100000//(2 * lsqr_iters + 1)

class EquilibratedLevMarNewCQR2Iter(EquilibratedNewCQR, LevMarNewCQR):
    """Equilibrated Lev Mar."""

    lsqr_iters = 2
    max_iterations = 100000//(2 * lsqr_iters + 1)


def l2_ruiz(matrix, d=None, e=None):
    """Simple Ruiz L2 equilibration."""

    def norm_cols(matrix):
        return np.linalg.norm(matrix, axis=0)

    def norm_rows(matrix):
        return np.linalg.norm(matrix, axis=1)

    m, n = matrix.shape
    if d is None:
        d = np.ones(m)
    if e is None:
        e = np.ones(n)

    work_matrix = ((matrix * e).T * d).T

    for i in range(1):

        nr = norm_rows(work_matrix)
        nc = norm_cols(work_matrix)

        r1 = np.max(nr[nr > 0]) / np.min(nr[nr > 0])
        r2 = np.max(nc[nc > 0]) / np.min(nc[nc > 0])
        # print(r1, r2)
        if (r1-1 < 1e-2) and (r2-1 < 1e-2):
            # logger.info('Equilibration converged.')
            break

        # print(r1, r2)

        d[nr > 0] *= nr[nr > 0]**(-0.5)
        e[nc > 0] *= ((m)/(n))**(0.25) * nc[nc > 0]**(-0.5)

        work_matrix = ((matrix * e).T * d).T
    return d, e


class PostEquilibratedLevMarNewCQR(EquilibratedNewCQR, LevMarNewCQR):
    """Idea with also diagonal equilibration of LevMar system."""

    lsqr_iters = 5
    max_iterations = 100000//(2 * lsqr_iters + 1)
    damp = 0.

    def prepare_loop(self):
        """Skip SOCs."""
        assert len(self.soc) == 0
        super().prepare_loop()

        mat = self.nullspace @ self.nullspace.T
        mat[np.abs(mat) < np.finfo(float).eps] = 0.
        self.base_mat = sp.sparse.csc_array(mat)
        self.post_d = np.ones(self.m)
        self.post_e = np.ones(self.m)

    def iterate(self):
        """Do one iteration."""

        mask = np.ones(self.m)
        mask[self.zero:] = self.z[self.zero:] > 0
        actual_mat = self.base_mat @ sp.sparse.diags(
            2 * mask - 1.) - sp.sparse.diags(mask)

        m = actual_mat.todense()
        self.post_d[:], self.post_e[:] = l2_ruiz(
            m, d=self.post_d, e=self.post_e)

        d = self.post_d
        e = self.post_e

        # why is this??
        assert np.allclose(d, e)
        internal_mat = sp.sparse.diags(d) @ actual_mat @ sp.sparse.diags(e)

        # breakpoint()

        self.y[:] = self.cone_project(self.z)
        step = self.linspace_project(2 * self.y - self.z) - self.y
        # print(np.linalg.norm(step))

        result = sp.sparse.linalg.lsqr(
                    internal_mat, -sp.sparse.diags(d) @ step,
                    x0=sp.sparse.diags(1./e) @ step,
                    damp=0., # might make sense to change this?
                    atol=0., btol=0., # might make sense to change this
                    iter_lim=self.lsqr_iters)
        # breakpoint()
        # print(result[1:-1])
        self.z[:] = self.z + sp.sparse.diags(e) @ result[0]

        # import matplotlib.pyplot as plt
        # breakpoint()
        # plt.plot(np.linalg.norm(m, axis=0))
        # plt.plot(np.linalg.norm(m, axis=1))
        # plt.show()

        # # breakpoint()

        # super().iterate()
