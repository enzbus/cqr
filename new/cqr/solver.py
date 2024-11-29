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
            self, matrix, b, c, zero, nonneg, x0=None, y0=None, qr='NUMPY',
            verbose=True):

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
        assert qr in ['NUMPY', 'PYSPQR']
        self.qr = qr
        self.verbose = verbose

        if self.verbose:
            print(
                f'Program: m={self.m}, n={self.n}, nnz={self.matrix.nnz},'
                f' zero={self.zero}, nonneg={self.nonneg}')

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

            # self.toy_solve()
            self.new_toy_solve()
            # self.x_transf, self.y = self.solve_program_cvxpy(
            #     self.matrix_qr_transf, b, self.c_qr_transf)
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
                    'zero': self.zero, 'nonneg': self.nonneg, 'second_order': []},
                max_iters=25)

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

        q, r, e = qr(self.matrix_ruiz_equil)
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
        # self.b_qr_transf = self.b_ruiz_equilb/self.sigma_qr
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

    def cone_project(self, s):
        """Project on program cone."""
        return np.concatenate([
            np.zeros(self.zero), np.maximum(s[self.zero:], 0.)])

    def dual_cone_project_basic(self, y):
        """Project on dual of program cone."""
        return np.concatenate([
            y[:self.zero], np.maximum(y[self.zero:], 0.)])

    def identity_minus_cone_project(self, s):
        """Identity minus projection on program cone."""
        return s - self.cone_project(s)

    def pri_err(self, x):
        """Error on primal cone."""
        s = self.b_qr_transf - self.matrix_qr_transf @ x
        return self.identity_minus_cone_project(s)

    def dual_cone_project_nozero(self, y):
        """Project on dual of program cone, skip zeros."""
        return np.maximum(y[self.zero:], 0.)

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

    def cone_project_derivative(self, s):
        """Derivative of projection on program cone."""
        if self.verbose:
            old_s_active = self.s_active if hasattr(
                self, 's_active') else np.ones(self.m-self.zero)
            self.s_active = 1. * (s[self.zero:] >= 0.)
            print('s_act_chgs=%d' % np.sum(
                np.abs(self.s_active - old_s_active)), end='\t')
        return sp.sparse.diags(
            np.concatenate([np.zeros(self.zero), 1 * (s[self.zero:] >= 0.)]))

    def identity_minus_cone_project_derivative(self, s):
        """Identity minus derivative of projection on program cone."""
        return sp.sparse.eye(self.m) - self.cone_project_derivative(s)

    def dual_cone_project_derivative_nozero(self, y):
        """Derivative of projection on dual of program cone, skip zeros."""
        if self.verbose:
            old_y_active = self.y_active if hasattr(
                self, 'y_active') else np.ones(self.m-self.zero)
            self.y_active = 1. * (y[self.zero:] >= 0.)
            print('y_act_chgs=%d' % np.sum(
                np.abs(self.y_active - old_y_active)), end='\t')
        return sp.sparse.diags(1 * (y[self.zero:] >= 0.))

    def identity_minus_dual_cone_project_derivative_nozero(self, y):
        """Identity minus derivative of projection on dual of program cone.

        (Skip zeros.)
        """
        return sp.sparse.eye(
            self.m - self.zero) - self.dual_cone_project_derivative_nozero(y)

    def newjacobian(self, var_reduced):
        """Jacobian of the residual using gap QR transform."""
        var = self.var0 + self.gap_NS @ var_reduced
        x = var[:self.n]
        y_reduced = var[self.n:]
        s = self.b_qr_transf - self.matrix_qr_transf @ x
        y = self.y0 + self.nullspace_projector @ y_reduced

        if self.m <= self.n:
            result = np.block(
                [[
                    -self.identity_minus_cone_project_derivative(
                        s) @ self.matrix_qr_transf]])
        else:
            result = np.block(
                [[-self.identity_minus_cone_project_derivative(
                    s) @ self.matrix_qr_transf,
                    np.zeros((self.m, self.m-self.n))],
                 [
                    np.zeros((self.m-self.zero, self.n)),
                    self.identity_minus_dual_cone_project_derivative_nozero(
                        y) @ self.nullspace_projector[self.zero:]],
                 ])

        # print('\n' *5)
        # print(np.linalg.svd(result @ self.gap_NS)[1])
        # print('\n' * 5)

        return result @ self.gap_NS

    def newjacobian_linop(self, var_reduced):
        """Jacobian of the residual function."""
        return self.coneproje_linop(var_reduced) @ self.newjacobian_linop_nocones()

    def coneproje_linop(self, var_reduced):
        """Jacobian of the cone projections.."""
        var = self.var0 + self.gap_NS @ var_reduced
        x = var[:self.n]
        s = self.b_qr_transf - self.matrix_qr_transf @ x
        s_derivative = self.identity_minus_cone_project_derivative(s)

        if self.m <= self.n:
            return sp.sparse.linalg.aslinearoperator(s_derivative)
        else:
            y_reduced = var[self.n:]
            y = self.y0 + self.nullspace_projector @ y_reduced
            y_derivative = self.identity_minus_dual_cone_project_derivative_nozero(
                y)
            return sp.sparse.linalg.aslinearoperator(sp.sparse.bmat([
                [s_derivative, None],
                [None, y_derivative]
            ]))

    def newjacobian_linop_nocones(self):
        """Linear component of the Jacobian of the residual function."""

        if self.m <= self.n:
            def matvec(dvar_reduced):
                return -(self.matrix_qr_transf @ (self.gap_NS @ dvar_reduced))

            def rmatvec(dres):
                return -self.gap_NS.T @ (self.matrix_qr_transf.T @ dres)
            return sp.sparse.linalg.LinearOperator(
                shape=(self.m, self.m-1),
                matvec=matvec,
                rmatvec=rmatvec,
            )
        else:
            def matvec(dvar_reduced):
                _ = self.gap_NS @ dvar_reduced
                dx = _[:self.n]
                dy_reduced = _[self.n:]
                dres0 = - (self.matrix_qr_transf @ dx)
                dres1 = (self.nullspace_projector @ dy_reduced)[self.zero:]
                return np.concatenate([dres0, dres1])

            def rmatvec(dres):
                dres0 = dres[:self.m]
                dres1 = dres[self.m:]
                dx = -(self.matrix_qr_transf.T  @ dres0)
                dy_reduced = self.nullspace_projector.T @ np.concatenate(
                    [np.zeros(self.zero), dres1])
                dvar_reduced = np.concatenate([dx, dy_reduced])
                return self.gap_NS.T @ dvar_reduced

            return sp.sparse.linalg.LinearOperator(
                shape=(2*self.m-self.zero, self.m-1),
                matvec=matvec,
                rmatvec=rmatvec,
            )

    ###
    # For Newton methods
    ###

    def newton_loss(self, var_reduced):
        """Loss used for Newton iterations."""
        return np.linalg.norm(self.newres(var_reduced)) ** 2 / 2.

    def newton_gradient(self, var_reduced):
        """Gradient used for Newton iterations."""
        return self.newjacobian_linop(var_reduced).T @ self.newres(var_reduced)

    def newton_hessian(self, var_reduced):
        """Hessian used for Newton iterations."""
        _jac = self.newjacobian_linop(var_reduced)
        return _jac.T @ _jac

    # @staticmethod
    def inexact_levemberg_marquardt(self,
                                    residual, jacobian, x0, max_iter=100000, max_ls=200, eps=1e-12,
                                    damp=0.):
        """Inexact Levemberg-Marquardt solver."""
        cur_x = np.copy(x0)
        cur_residual = residual(cur_x)
        cur_loss = np.linalg.norm(cur_residual)
        cur_jacobian = jacobian(cur_x)
        TOTAL_CG_ITER = 0

        def _counter(_):
            nonlocal TOTAL_CG_ITER
            TOTAL_CG_ITER += 1
        TOTAL_BACK_TRACKS = 0
        for i in range(max_iter):
            if self.verbose:
                print("it=%d" % i, end='\t')
                print("cvx_loss=%.2e" % np.linalg.norm(
                    self.newres(cur_x)), end='\t')
                print("ref_loss=%.2e" % np.linalg.norm(
                    self.refinement_residual(cur_x)), end='\t')
            cur_gradient = cur_jacobian.T @ cur_residual
            cur_hessian = cur_jacobian.T @ cur_jacobian
            # in solver_new I was doing extra regularization inside
            # the cone projection in between the two residual jacobian;
            # with new formulation this should have same effect
            if damp > 0.:
                cur_hessian += sp.sparse.linalg.aslinearoperator(
                    sp.sparse.eye(len(cur_gradient)) * damp)

            # fallback for Scipy < 1.12 doesn't work; forcing >= 1.12 for
            # now, I won't use this function anyway
            # sp_version = [int(el) for el in sp.__version__.split('.')]
            # if sp_version >= [1,12]:
            olditers = int(TOTAL_CG_ITER)
            _ = sp.sparse.linalg.cg(
                A=cur_hessian,
                b=-cur_gradient,
                rtol=min(0.5, np.linalg.norm(cur_gradient)**0.5),
                callback=_counter,
                # maxiter=30,
            )
            if self.verbose:
                print('cg_iters=%d' % (TOTAL_CG_ITER - olditers), end='\t')
            # else:
            #     _ = sp.sparse.linalg.cg(
            #         A = cur_hessian,
            #         b = -cur_gradient,
            #         tol= min(0.5, np.linalg.norm(cur_gradient)**0.5),
            #         callback=_counter,
            #     )
            step = _[0]
            for j in range(max_ls):
                step_len = 0.9**j
                new_x = cur_x + step * step_len
                new_residual = residual(new_x)
                new_loss = np.linalg.norm(new_residual)
                if new_loss < cur_loss:
                    cur_x = new_x
                    cur_residual = new_residual
                    cur_loss = new_loss
                    if self.verbose:
                        print(f'btrcks={j}', end='\n')
                    TOTAL_BACK_TRACKS += j
                    break
            else:
                if self.verbose:
                    print(
                        'Line search failed, exiting.'
                    )
                break
            # convergence check
            cur_jacobian = jacobian(cur_x)
            cur_gradient = cur_jacobian.T @ cur_residual

            if np.max(np.abs(cur_gradient)) < eps:
                if self.verbose:
                    print(
                        'Converged, cur_gradient norm_inf=%.2e' %
                        np.max(np.abs(cur_gradient)))
                break
        print('iters', i)
        print('total CG iters', TOTAL_CG_ITER)
        print('total backtracks', TOTAL_BACK_TRACKS)
        return np.array(cur_x, dtype=float)

    def refinement_residual(self, var_reduced):
        """Residual for refinement."""

        var = self.var0 + self.gap_NS @ var_reduced
        x_transf = var[:self.n]
        s = self.b_qr_transf - self.matrix_qr_transf @ x_transf
        y_reduced = var[self.n:]
        y = self.y0 + self.nullspace_projector @ y_reduced
        return self.dual_cone_project_basic(y - s) - y

    def refinement_jacobian(self, var_reduced):
        """Jacobian of the refinement residual."""

        # TODO: consider also other case
        assert self.m > self.n

        # for now dense only
        assert self.qr == 'NUMPY'

        var = self.var0 + self.gap_NS @ var_reduced
        x_transf = var[:self.n]
        s = self.b_qr_transf - self.matrix_qr_transf @ x_transf
        y_reduced = var[self.n:]
        y = self.y0 + self.nullspace_projector @ y_reduced
        z = y-s
        z_derivative_nozero = self.dual_cone_project_derivative_nozero(z)

        matrix1 = np.hstack([self.matrix_qr_transf, self.nullspace_projector])
        matrix1[self.zero:] = z_derivative_nozero @ matrix1[self.zero:]
        matrix2 = np.hstack(
            [np.zeros((self.m, self.n)), self.nullspace_projector])
        return (matrix1 - matrix2) @ self.gap_NS

    def _refine(self):
        """Refine with new formulation."""
        self.var_reduced = self.inexact_levemberg_marquardt(
            self.refinement_residual, self.refinement_jacobian,
            self.var_reduced, eps=1e-15)

    def refine(self):
        """Basic refinement."""

        print('Refinement loss at end of main loop',
              np.linalg.norm(self.refinement_residual(self.var_reduced)))

        self._refine()
        # self._refine()
        # self._refine()

        print('Refinement loss after refine',
              np.linalg.norm(self.refinement_residual(self.var_reduced)))

    def new_toy_solve(self):
        """Solve by LM."""

        # breakpoint()
        # _ = self.inexact_levemberg_marquardt(self.refinement_residual, self.refinement_jacobian, self.var_reduced, eps=1e-15, max_iter=10)
        # self.var_reduced = self.inexact_levemberg_marquardt(self.newres, self.newjacobian_linop, self.var_reduced, max_iter=10)
        # breakpoint()

        self.var_reduced = self.inexact_levemberg_marquardt(
            self.newres, self.newjacobian_linop, self.var_reduced)

        # breakpoint()
        # for i in range(10):
        #     print( sp.optimize.check_grad(self.refinement_residual, self.refinement_jacobian, np.random.randn(self.m-1)))
        # breakpoint()

        # J = self.refinement_jacobian(self.var_reduced); r = self.refinement_residual(self.var_reduced); step = sp.sparse.linalg.lsqr(J, -r)[0]
        # breakpoint()

    def old_toy_solve(self):
        result = sp.optimize.least_squares(
            self.newres, np.zeros(self.m-1),
            jac=self.newjacobian, method='lm',
            ftol=1e-15, xtol=1e-15, gtol=1e-15,)
        print(result)
        self.var_reduced = result.x

        # opt_loss = result.cost

        # result = sp.optimize.fmin_ncg(
        #     f=self.newton_loss,
        #     x0=np.zeros(self.m-1),
        #     fprime=self.newton_gradient,
        #     fhess=self.newton_hessian,
        #     disp=True,
        #     full_output=True,
        #     avextol=1e-16,
        #     #self.newres, np.zeros(self.m-1),
        #     #jac=self.newjacobian, method='lm',
        #     #ftol=1e-15, xtol=1e-15, gtol=1e-15,
        #     )
        # print(result)

        # opt_var_reduced = result[0]
        # opt_loss = result[1]

        # exit(0)

        # result = sp.optimize.fmin_l_bfgs_b(
        #     func=self.newton_loss,
        #     x0=np.zeros(self.m-1),
        #     fprime=self.newton_gradient,
        #     # fhess=self.newton_hessian,
        #     # disp=True,
        #     factr=0.1,
        #     pgtol=1e-16,
        #     # full_output=True,
        #     # avextol=1e-16,
        #     #self.newres, np.zeros(self.m-1),
        #     #jac=self.newjacobian, method='lm',
        #     #ftol=1e-15, xtol=1e-15, gtol=1e-15,
        #     )
        # print(result)

        # opt_var_reduced = result[0]
        # opt_loss = result[1]

    def decide_solution_or_certificate(self):
        """Decide if solution or certificate."""

        residual = self.newres(self.var_reduced)
        sqloss = np.linalg.norm(residual)**2/2.

        print("sq norm of residual", sqloss)
        print("sq norm of jac times residual",
              np.linalg.norm(self.newjacobian_linop(self.var_reduced).T @ residual)**2/2.)

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
            if self.qr == 'NUMPY' and self.m > self.n:
                self.refine()
