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
"""Solver class.

Idea:

Centralizes memory allocation, its managed memory translates to a struct in C.
Each method, which should be very small and simple, translates to a C function.
Experiments (new features, ...) should be done as subclasses.
"""

# import cvxpy as cp
import numpy as np
import scipy as sp

from .equilibrate import hsde_ruiz_equilibration
# from .line_search import LineSearcher, LineSearchFailed

from pyspqr import qr


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
    :param x0: Initial guess of the primal variable. Default None,
        equivalent to zero vector.
    :type x0: np.array or None.
    :param y0: Initial guess of the dual variable. Default None,
        equivalent to zero vector.
    :type y0: np.array or None.
    """

    def __init__(
            self, matrix, b, c, zero, nonneg, soc=(), x0=None, y0=None,
            qr='PYSPQR', verbose=True):

        # process program data
        self.matrix = sp.sparse.csc_matrix(matrix)
        self.m, self.n = matrix.shape
        assert zero >= 0
        assert nonneg >= 0
        for soc_dim in soc:
            assert soc_dim > 1
        assert zero + nonneg + sum(soc) == self.m
        self.zero = zero
        self.nonneg = nonneg
        self.soc = soc
        assert len(b) == self.m
        self.b = np.array(b, dtype=float)
        assert len(c) == self.n
        self.c = np.array(c, dtype=float)
        assert qr in ['NUMPY', 'PYSPQR']
        self.qr = qr
        self.verbose = verbose

        if self.verbose:
            print(
                f'Program: m={self.m}, n={self.n}, nnz={self.matrix.nnz},'
                f' zero={self.zero}, nonneg={self.nonneg}, soc={self.soc}')

        self.x = np.zeros(self.n) if x0 is None else np.array(x0)
        assert len(self.x) == self.n
        self.y = np.zeros(self.m) if y0 is None else np.array(y0)
        assert len(self.y) == self.m

        # self.y = np.empty(self.m, dtype=float)
        # self.update_variables(x0=x0, y0=y0)

        try:
            self._equilibrate()
            self._qr_transform_program_data()
            self._qr_transform_dual_space()
            self._qr_transform_gap()

            self.admm_intercept = self.admm_linspace_project(np.zeros(self.m*2))

            #### self.toy_solve()
            ##### self.x_transf, self.y = self.solve_program_cvxpy(
            #####     self.matrix_qr_transf, b, self.c_qr_transf)

            # self.new_toy_solve()
            # self.var_reduced = self.toy_admm_solve(self.var_reduced)
            # self.var_reduced = self.old_toy_douglas_rachford_solve(self.var_reduced)

            # self.decide_solution_or_certificate()
            # self.toy_douglas_rachford_solve()
            self.new_toy_douglas_rachford_solve()
            self.decide_solution_or_certificate()

            self._invert_qr_transform_gap()
            self._invert_qr_transform_dual_space()
            self._invert_qr_transform()
            self.status = 'Optimal'
        except Infeasible:
            self.status = 'Infeasible'
        except Unbounded:
            self._invert_qr_transform()
            self.status = 'Unbounded'

        self._invert_equilibrate()

        print('Resulting status:', self.status)

    def backsolve_r(self, vector, transpose=True):
        """Simple triangular solve with matrix R."""
        if transpose:  # forward transform c
            r = self.r.T
        else:  # backward tranform x
            r = self.r

        # TODO: handle all degeneracies here
        # try:
        #     result = sp.linalg.solve_triangular(r, vector, lower=transpose)
        #     ...
        # except np.linalg.LinAlgError:
        #

        # TODO: this case can be handled much more efficiently
        result = np.linalg.lstsq(r, vector, rcond=None)[0]

        if not np.allclose(r @ result, vector):
            if transpose:
                # TODO: make sure this tested, what do we need to set on exit?
                raise Unbounded(
                    "Cost vector is not in the span of the program matrix!")
            else:
                # TODO: figure out when this happens
                raise Exception('Solver error.')
        return result

    # def update_variables(self, x0=None, y0=None):
    #     """Update initial values of the primal and dual variables.

    #     :param x0: Initial guess of the primal variable. Default None,
    #         equivalent to zero vector.
    #     :type x0: np.array or None.
    #     :param y0: Initial guess of the dual variable. Default None,
    #         equivalent to zero vector.
    #     :type y0: np.array or None.
    #     """

    #     if x0 is None:
    #         self.x[:] = np.zeros(self.n, dtype=float)
    #     else:
    #         assert len(x0) == self.n
    #         self.x[:] = np.array(x0, dtype=float)
    #     if y0 is None:
    #         self.y[:] = np.zeros(self.m, dtype=float)
    #     else:
    #         assert len(y0) == self.m
    #         self.y[:] = np.array(y0, dtype=float)

    def _equilibrate(self):
        """Apply Ruiz equilibration to program data."""
        self.equil_d, self.equil_e, self.equil_sigma, self.equil_rho, \
            self.matrix_ruiz_equil, self.b_ruiz_equil, self.c_ruiz_equil = \
            hsde_ruiz_equilibration(
                self.matrix, self.b, self.c, dimensions={
                    'zero': self.zero, 'nonneg': self.nonneg, 'second_order': self.soc},
                max_iters=5, l_norm=2, eps_cols=1e-12, eps_rows=1e-12)

        self.x_equil = self.equil_sigma * (self.x / self.equil_e)
        self.y_equil = self.equil_rho * (self.y / self.equil_d)

    def _invert_equilibrate(self):
        """Invert Ruiz equlibration."""
        # TODO: make sure with certificates you always return something
        x_equil = self.x_equil if hasattr(
            self, 'x_equil') else np.zeros(self.n)
        y_equil = self.y_equil if hasattr(
            self, 'y_equil') else np.zeros(self.m)

        self.x = (self.equil_e * x_equil) / self.equil_sigma
        self.y = (self.equil_d * y_equil) / self.equil_rho

    def _qr_transform_program_data_pyspqr(self):
        """Apply QR decomposition to equilibrated program data."""

        q, r, e = qr(self.matrix_ruiz_equil, ordering='AMD')
        shape1 = min(self.n, self.m)
        self.matrix_qr_transf = sp.sparse.linalg.LinearOperator(
            shape=(self.m, shape1),
            matvec=lambda x: q @ np.concatenate([x, np.zeros(self.m-shape1)]),
            rmatvec=lambda y: (
                q.T @ np.array(y, copy=True).reshape(y.size))[:shape1],
        )
        shape2 = max(self.m - self.n, 0)
        self.nullspace_projector = sp.sparse.linalg.LinearOperator(
            shape=(self.m, shape2),
            matvec=lambda x: q @ np.concatenate([np.zeros(self.m-shape2), x]),
            rmatvec=lambda y: (
                q.T @ np.array(y, copy=True).reshape(y.size))[self.m-shape2:]
        )
        self.r = (r.todense() @ e)[:self.n]

    def _qr_transform_program_data_numpy(self):
        """Apply QR decomposition to equilibrated program data."""

        q, r = np.linalg.qr(self.matrix_ruiz_equil.todense(), mode='complete')
        self.matrix_qr_transf = q[:, :self.n].A
        self.nullspace_projector = q[:, self.n:].A
        self.r = r[:self.n].A

    def _qr_transform_program_data(self):
        """Delegate to either Numpy or PySPQR, create constants."""
        if self.qr == 'NUMPY':
            self._qr_transform_program_data_numpy()
        elif self.qr == 'PYSPQR':
            self._qr_transform_program_data_pyspqr()
        else:
            raise SyntaxError('Wrong qr setting!')

        self.c_qr_transf = self.backsolve_r(self.c_ruiz_equil)

        # TODO: unclear if this helps
        # self.sigma_qr = np.linalg.norm(self.b_ruiz_equil)
        # self.b_qr_transf = self.b_ruiz_equil/self.sigma_qr
        self.sigma_qr = 1.
        self.b_qr_transf = self.b_ruiz_equil

        # TODO: what happens in degenerate cases here?
        self.x_transf = self.r @ (self.x_equil / self.sigma_qr)

    def _invert_qr_transform(self):
        """Simple triangular solve with matrix R."""
        result = self.backsolve_r(
            vector=self.x_transf, transpose=False)
        self.x_equil = result * self.sigma_qr

    def _qr_transform_dual_space(self):
        """Apply QR transformation to dual space."""
        self.y0 = self.matrix_qr_transf @ -self.c_qr_transf
        if self.m <= self.n:
            if not np.allclose(
                    self.dual_cone_project_basic(self.y0),
                    self.y0):

                # TODO: double check this logic
                s_certificate = self.cone_project(-self.y0)
                self.x_transf = - self.matrix_qr_transf.T @ s_certificate
                # print('Unboundedness certificate', self.x)
                raise Unbounded("There is no feasible dual vector.")
        # diff = self.y - self.y0
        # self.y_reduced = self.nullspace_projector.T @ diff
        self.b0 = self.b_qr_transf @ self.y0
        self.b_reduced = self.b_qr_transf @ self.nullspace_projector

        # propagate y_equil
        self.y_reduced = self.nullspace_projector.T @ self.y_equil

    def _invert_qr_transform_dual_space(self):
        """Invert QR transformation of dual space."""
        self.y_equil = self.y0 + self.nullspace_projector @ self.y_reduced

    def _qr_transform_gap_pyspqr(self):
        """Apply QR transformation to zero-gap residual."""
        mat = np.concatenate([
            self.c_qr_transf, self.b_reduced]).reshape((self.m, 1))
        mat = sp.sparse.csc_matrix(mat)
        q, r, e = qr(mat)

        self.gap_NS = sp.sparse.linalg.LinearOperator(
            shape=(self.m, self.m-1),
            matvec=lambda var_reduced: q @ np.concatenate(
                [[0.], var_reduced]),
            rmatvec=lambda var: (q.T @ var)[1:]
        )

    def _qr_transform_gap_numpy(self):
        """Apply QR transformation to zero-gap residual."""
        Q, R = np.linalg.qr(
            np.concatenate(
                [self.c_qr_transf, self.b_reduced]).reshape((self.m, 1)),
            mode='complete')
        self.gap_NS = Q[:, 1:]

    def _qr_transform_gap(self):
        """Delegate to either Numpy or PySPQR, create constants."""
        if self.qr == 'NUMPY':
            self._qr_transform_gap_numpy()
        elif self.qr == 'PYSPQR':
            self._qr_transform_gap_pyspqr()
        else:
            raise SyntaxError('Wrong qr setting!')

        self.var0 = - self.b0 * np.concatenate(
            [self.c_qr_transf, self.b_reduced]) / np.linalg.norm(
                np.concatenate([self.c_qr_transf, self.b_reduced]))**2

        # propagate x_transf and y_reduced
        var = np.concatenate([self.x_transf, self.y_reduced])
        self.var_reduced = self.gap_NS.T @ var

    def _invert_qr_transform_gap(self):
        """Invert QR transformation of zero-gap residual."""
        var = self.var0 + self.gap_NS @ self.var_reduced
        self.x_transf = var[:self.n]
        self.y_reduced = var[self.n:]

    @staticmethod
    def second_order_project(z, result):
        """Project on second-order cone.

        :param z: Input array.
        :type z: np.array
        :param result: Resulting array.
        :type result: np.array
        """

        assert len(z) >= 2

        y, t = z[1:], z[0]

        # cache this?
        norm_y = np.linalg.norm(y)

        if norm_y <= t:
            result[:] = z
            return

        if norm_y <= -t:
            result[:] = 0.
            return

        result[0] = 1.
        result[1:] = y / norm_y
        result *= (norm_y + t) / 2.

    @staticmethod
    def new_second_order_project(z, pi, two_pi_minus_z):
        """Project on second-order cone.

        :param z: Input array.
        :type z: np.array
        :param result: Resulting array.
        :type result: np.array
        """

        assert len(z) >= 2

        y, t = z[1:], z[0]

        # cache this?
        norm_y = np.linalg.norm(y)

        if norm_y <= t:
            pi[:] = z
            two_pi_minus_z[:] = z
            return

        if norm_y <= -t:
            pi[:] = 0.
            two_pi_minus_z[:] = -z
            return

        pi[0] = (norm_y + t) / 2.
        two_pi_minus_z[0] = norm_y
        pi[1:] = ((1. + t/norm_y) / 2.) * y
        two_pi_minus_z[1:] = (t/norm_y) * y

    def self_dual_cone_project(self, conic_var):
        """Project on self-dual cones."""
        result = np.empty_like(conic_var)
        result[:self.nonneg] = np.maximum(conic_var[:self.nonneg], 0.)
        cur = self.nonneg
        for soc_dim in self.soc:
            self.second_order_project(
                conic_var[cur:cur+soc_dim], result[cur:cur+soc_dim])
            cur += soc_dim
        return result

    def cone_project(self, s):
        """Project on program cone."""
        return np.concatenate([
            np.zeros(self.zero), self.self_dual_cone_project(s[self.zero:])])

    def dual_cone_project_basic(self, y):
        """Project on dual of program cone."""
        return np.concatenate([
            y[:self.zero], self.self_dual_cone_project(y[self.zero:])])

    ##
    # ADMM Idea
    ##

    def admm_cone_project(self, sy):
        """Project ADMM variable on the cone."""
        s = sy[:self.m]
        y = sy[self.m:]
        pi_s = self.cone_project(s)
        pi_y = self.dual_cone_project_basic(y)
        return np.concatenate([pi_s, pi_y])

    def new_admm_cone_project(self, sy):
        """Project ADMM variable on the cone."""
        s = sy[:self.m]
        y = sy[self.m:]
        pi = np.empty(self.m*2)
        two_pi_minus_sy = np.empty(self.m*2)

        cur = 0

        # s, zero cone
        pi[:self.zero] = np.zeros(self.zero)
        two_pi_minus_sy[:self.zero] = -s[:self.zero]

        cur += self.zero

        # s, nonneg cone
        pi[cur:cur+self.nonneg] = np.maximum(s[cur:cur+self.nonneg], 0.)
        two_pi_minus_sy[cur:cur+self.nonneg] = 2 * pi[cur:cur+self.nonneg] - s[cur:cur+self.nonneg]

        cur += self.nonneg

        # s, soc cones
        for q in self.soc:
            self.new_second_order_project(s[cur:cur+q], pi[cur:cur+q], two_pi_minus_sy[cur:cur+q])
            cur += q

        # y, zero cone
        pi[cur:cur+self.zero] = y[:self.zero]
        two_pi_minus_sy[cur:cur+self.zero] = y[:self.zero]

        cur += self.zero

        # y, nonneg cone
        pi[cur:cur+self.nonneg] = np.maximum(y[cur-self.m:cur-self.m+self.nonneg], 0.)
        two_pi_minus_sy[cur:cur+self.nonneg] = 2 * pi[cur:cur+self.nonneg] - y[cur-self.m:cur-self.m+self.nonneg]

        cur += self.nonneg

        # y, soc cones
        for q in self.soc:
            self.new_second_order_project(y[cur-self.m:cur-self.m+q], pi[cur:cur+q], two_pi_minus_sy[cur:cur+q])
            # two_pi_minus_sy[cur:cur+q] = 2 * pi[cur:cur+q] - y[cur-self.m:cur-self.m+q]
            cur += q

        assert cur == self.m * 2

        return pi, two_pi_minus_sy

    def _sy_from_var_reduced(self, var_reduced):
        """Get sy from var reduced."""
        var = self.var0 + self.gap_NS @ var_reduced
        s = self.b_qr_transf - self.matrix_qr_transf @ var[:self.n]
        y = self.y0 + self.nullspace_projector @ var[self.n:]
        return np.concatenate([s, y])

    def _var_reduced_from_sy(self, sy):
        """Get var reduced from sy in least squares sense."""
        s = sy[:self.m]
        y = sy[self.m:]
        var1 = self.matrix_qr_transf.T  @ (self.b_qr_transf - s)
        var2 = self.nullspace_projector.T @ (y - self.y0)
        var = np.concatenate([var1, var2])
        return self.gap_NS.T @ (var - self.var0)

    def _sy_from_var_reduced_noconst(self, var_reduced):
        """Get sy from var reduced, w/out constants."""
        var = self.gap_NS @ var_reduced
        s = -self.matrix_qr_transf @ var[:self.n]
        y = self.nullspace_projector @ var[self.n:]
        return np.concatenate([s, y])

    def _var_reduced_from_sy_noconst(self, sy):
        """Get var reduced from sy in least squares sense, w/out constants."""
        s = sy[:self.m]
        y = sy[self.m:]
        var1 = self.matrix_qr_transf.T  @ (- s)
        var2 = self.nullspace_projector.T @ (y)
        var = np.concatenate([var1, var2])
        return self.gap_NS.T @ (var)

    def admm_linspace_project(self, sy):
        """Project ADMM variable on the subspace."""
        vr = self._var_reduced_from_sy(sy)
        return self._sy_from_var_reduced(vr)

    def admm_linspace_project_noconst(self, sy):
        """Project ADMM variable on the subspace."""
        vr = self._var_reduced_from_sy_noconst(sy)
        return self._sy_from_var_reduced_noconst(vr)

    def douglas_rachford_step(self, dr_y):
        """Douglas-Rachford step.

        https://www.seas.ucla.edu/~vandenbe/236C/lectures/dr.pdf,
        slides 11.2-3.
        """
        # self.admm_linspace_project(2 * self.admm_cone_project(dr_y) - dr_y) - self.admm_cone_project(dr_y)
        pi, two_pi_minus_sy = self.new_admm_cone_project(dr_y)
        # tmp = self.admm_cone_project(dr_y)
        # assert np.allclose(tmp, pi)
        # if not hasattr(self, "admm_intercept"):
        # return self.admm_intercept + self.admm_linspace_project_noconst(2 * tmp - dr_y) - tmp
        # return self.admm_linspace_project(2 * tmp - dr_y) - tmp
        return self.admm_linspace_project(two_pi_minus_sy) - pi
        # else:
        #     return self.admm_linspace_project_ex_intercept(2 * tmp - dr_y) - tmp

        # tmp = self.admm_linspace_project(dr_y)
        # return self.admm_cone_project(2 * tmp - dr_y) - tmp

        # return self.admm_linspace_project(self.admm_cone_project(dr_y)) - dr_y

    def douglas_rachford_step_derivative(self, dr_y):
        """Douglas-Rachford step derivative.

        Note that it is not symmetric! Transpose is the same as
        switching the 2 projections; that's why it performs the same
        if you switch them.
        """

        dpicone = self.admm_cone_project_derivative(dr_y)
        dpilin = self.admm_linspace_project_derivative()

        def matvec(dr_dy):
            tmp = dpicone @ dr_dy
            return dpilin @ (2 * tmp - dr_dy) - tmp

        def rmatvec(dr_df):
            tmp = dpilin @ dr_df
            return dpicone @ (2 * tmp - dr_df) - tmp

        return sp.sparse.linalg.LinearOperator(
            shape=(2 * self.m, 2 * self.m),
            dtype=float,
            matvec=matvec,
            rmatvec=rmatvec)

    def new_toy_douglas_rachford_solve(self, max_iter=int(1e5), eps=1e-12):
        """Simple Douglas-Rachford iteration."""
        dr_y = self._sy_from_var_reduced(self.var_reduced)
        # self.admm_compute_intercept()

        losses = []
        # steps = []
        # xs = []
        # breakpoint()

        ##
        # Analize vectors
        ##

        # for var in [self.var0, self.y0, self.b_qr_transf, self.admm_intercept]:
        #     _ = np.abs(var)
        #     _ = _[_ > 10 * np.finfo(float).eps]
        #     print(f'min={np.min(_):.2e} max={np.max(_):.2e}')

        # breakpoint()

        for i in range(max_iter):
            step = self.douglas_rachford_step(dr_y)
            losses.append(np.linalg.norm(step))
            # xs.append(dr_y)
            # steps.append(step)
            # print(f'iter {i} loss {losses[-1]:.2e}')
            if np.linalg.norm(step) < eps:
                print(f'converged in {i} iterations')
                break

            dr_y = np.copy(dr_y + step)

            # infeas / unbound
            if i % 100 == 99:
                tmp = self.admm_linspace_project(dr_y)
                cert = tmp - self.admm_cone_project(tmp)
                # y_cert = cert[:self.m]
                # s_cert = cert[self.m:]
                # x_cert = self.matrix_qr_transf.T @ cert[self.m:]
                cert /= np.linalg.norm(cert) # no, shoud normalize y by b and x,s by c
                # TODO double check this logic
                if (np.linalg.norm(self.matrix_qr_transf.T @ cert[:self.m]) < eps) and (np.linalg.norm(self.matrix_qr_transf @ self.matrix_qr_transf.T @ cert[self.m:] - cert[self.m:]) < eps):
                    # print('INFEASIBLE')
                    break

        else: # TODO: needs early stopping for infeas/unbound

            import matplotlib.pyplot as plt
            plt.semilogy(losses)
            plt.show()

            raise NotImplementedError

        self.var_reduced = self._var_reduced_from_sy(
            self.admm_cone_project(dr_y))
        print('SQNORM RESIDUAL OF SOLUTION',
            np.linalg.norm(self.newres(self.var_reduced))**2)

        # import matplotlib.pyplot as plt
        # plt.semilogy(losses)
        # plt.show()

    def identity_minus_cone_project(self, s):
        """Identity minus projection on program cone."""
        return s - self.cone_project(s)

    def pri_err(self, x):
        """Error on primal cone."""
        s = self.b_qr_transf - self.matrix_qr_transf @ x
        return self.identity_minus_cone_project(s)

    def dual_cone_project_nozero(self, y):
        """Project on dual of program cone, skip zeros."""
        return self.self_dual_cone_project(y[self.zero:])

    def identity_minus_dual_cone_project_nozero(self, y):
        """Identity minus projection on dual of program cone, skip zeros."""
        return y[self.zero:] - self.dual_cone_project_nozero(y)

    def dua_err(self, y_reduced):
        """Error on dual cone."""
        y = self.y0 + self.nullspace_projector @ y_reduced
        return self.identity_minus_dual_cone_project_nozero(y)

    def newres(self, var_reduced):
        """Residual using gap QR transform."""
        var = self.var0 + self.gap_NS @ var_reduced
        x = var[:self.n]
        y_reduced = var[self.n:]
        if self.m <= self.n:
            return self.pri_err(x)
        return np.concatenate(
            [self.pri_err(x), self.dua_err(y_reduced)])

    def decide_solution_or_certificate(self):
        """Decide if solution or certificate."""

        residual = self.newres(self.var_reduced)
        sqloss = np.linalg.norm(residual)**2/2.

        print("sq norm of residual", sqloss)
        # print("sq norm of jac times residual",
        #       np.linalg.norm(self.newjacobian_linop(self.var_reduced).T @ residual)**2/2.)

        if sqloss > 1e-12:
            # infeasible; for convenience we just set this here,
            # will have to check which is valid and maybe throw exceptions
            self.y_equil = -residual[:self.m]
            if np.linalg.norm(self.y_equil)**2 > 1e-12:
                # print('infeasibility certificate')
                # print(self.y_equil)
                raise Infeasible()

            s_certificate = -residual[self.m:]
            if self.zero > 0:
                s_certificate = np.concatenate(
                    [np.zeros(self.zero), s_certificate])
            if np.linalg.norm(s_certificate)**2 > 1e-12:
                # print('unboundedness certificate')
                self.x_transf = - self.matrix_qr_transf.T @ s_certificate
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

        else:  # for now we only refine solutions
            if self.m > self.n:
                pass
                # self.refine()
