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
"""Probably last rewriting; figure out correct Br regularization.
"""

import numpy as np
import scipy as sp
from ..base_solver import BaseSolver
from pyspqr import qr

class NewNewNewCQR(BaseSolver):
    """New idea for base CQR formulation."""

    max_iterations = 100000

    used_matrix = "matrix"
    used_b = "b"
    used_c = "c"
    use_numpy = True
    pd_scale = 1.0

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

        if self.use_numpy:
            self.c_qr = sp.linalg.solve_triangular(
                self.triangular.T, getattr(self, self.used_c), lower=True)
        else:
            self.c_qr = sp.sparse.linalg.spsolve_triangular(
                self.pyspqr_r.T, self.pyspqr_e @ getattr(
                    self, self.used_c), lower=True)

        self.z = np.zeros(self.m)
        self.y = np.zeros(self.m)
        self.s = np.zeros(self.m)
        self.x = np.zeros(self.n)
        self.pri_res = np.zeros(self.m)
        self.dua_res = np.zeros(self.m)

    def cone_project(self, z):
        """Project on y cone."""
        return self.composed_cone_project(
            z, has_zero=False, has_free=True, has_hsde=False)

    def compute_pridual_step(self, z):
        """Compute primal dual things, all descaled; steps are the residuals.
        
        So, in basic DR
        newz = y + dua_step + pd_scale * (-s + pri_step)
        
        """
        y = self.cone_project(z)
        s = y - z
        s /= self.pd_scale
        pri_step = self.nullspace @ (self.nullspace.T @ (s - getattr(self, self.used_b)))
        dua_step = -self.qr_matrix @ (self.qr_matrix.T @ y + self.c_qr)
        return s, y, pri_step, dua_step

    def iterate(self):
        """Simple Douglas Rachford iteration."""
        # compute primal-dual things, DR step is obtained from them
        self.s[:], self.y[:], self.pri_res[:], self.dua_res[:] = \
            self.compute_pridual_step(self.z)

        self.z[:] = (self.y + self.dua_res[:]) + self.pd_scale * (-self.s[:] + self.pri_res)

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        # this projection is probably superflous but it's just for diagnostics
        self.y[:] = self.cone_project(self.z)
        self.s[:] = (self.y - self.z) / self.pd_scale
        x_qr = self.qr_matrix.T @ (getattr(self, self.used_b) - self.s)
        if self.use_numpy:
            self.x[:] = sp.linalg.solve_triangular(self.triangular, x_qr, lower=False)
        else:
            self.x[:] = self.pyspqr_e.T @ sp.sparse.linalg.spsolve_triangular(
                self.pyspqr_r, x_qr, lower=False)

class EquilibratedNewNewNewCQR(NewNewNewCQR):
    """With Ruiz equilibration."""

    # max_iterations = 1000

    used_matrix = "eq_matrix"
    used_b = "eq_b"
    used_c = "eq_c"
    ruiz_max_rounds = 100
    ruiz_col_limit = 0.1 # converge if max norms cols <= 1.1 min norms cols
    ruiz_row_limit = 0.1 # converge if max norms rows <= 1.1 min norms rows
    ruiz_norm = 2
    do_soc_equalization = True
    internal_verbose = True

    def prepare_loop(self):
        """Do Ruiz equilibration."""

        matrix = self.matrix.todense()
        concatenated = np.block(
            [[matrix, self.b.reshape(self.m, 1)],
            [self.c.reshape(1, self.n), np.zeros((1, 1))]])
        if hasattr(concatenated, "A"):
            concatenated = concatenated.A
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

        for _ in range(self.ruiz_max_rounds):

            nr = norm_rows(work_matrix)
            nc = norm_cols(work_matrix)

            # equalize nr for SOCs
            cur = self.zero + self.nonneg
            for soc_dim in self.soc:
                if self.do_soc_equalization:
                    if self.ruiz_norm == np.inf:
                        nr[cur:cur+soc_dim] = np.max(nr[cur:cur+soc_dim])
                    elif self.ruiz_norm == 2:
                        nr[cur:cur+soc_dim] = np.sqrt(
                            np.mean(nr[cur:cur+soc_dim]**2))
                    else:
                        raise SyntaxError
                cur += soc_dim

            r1 = max(nr[nr > 0]) / min(nr[nr > 0])
            r2 = max(nc[nc > 0]) / min(nc[nc > 0])
            if self.internal_verbose:
                print("Ruiz ratio rows", r1, "Ruiz ratio cols", r2)
            if (r1-1 < self.ruiz_row_limit) and (r2-1 < self.ruiz_col_limit):
                break
            d_and_rho[nr > 0] *= nr[nr > 0]**(-0.5)
            e_and_sigma[nc > 0] *= ((m+1)/(n+1))**(1./(2 * self.ruiz_norm)) * nc[nc > 0]**(-0.5)

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


class BroydenEqNNNCQR(EquilibratedNewNewNewCQR):
    """Redo base class; refactored Broyden2EqNNCqr with minor fixes.

    Work in progress, best test so far 2025-10-25.

    Main changes: rewritten Broyden loop with full primal dual separation.
    Doesn't seem to have big effect but does probably improve a bit numerical
    accuracy. It costs one more dot and 2 more vector sums per Broyden step
    (so, many) so need to figure out if really useful. Changed Broyden
    regularization to something which probably is really quadratic
    regularization, seems to work with higher cap. Increasing the cap makes it
    better convergent on the portfolio_bad class. It might be that original
    flat cap works as well, with higher cap limits. Thing missing is loop to
    increase regularization if step len doesn't improve, online. 
    """

    memory = 50
    max_iterations = 100_000

    # basic regularization scheme
    broyden_regularizer = 1.0
    broyden_regularizer_increment = 2.0
    broyden_regularizer_decrement = broyden_regularizer_increment**(-1./memory)
    broyden_regularizer_ceil = 1.
    broyden_regularizer_floor = 1e-8

    use_numpy = False

    # PID
    Kp = 0.25
    Ki = 0.15
    Kd = 0.075

    internal_verbose = True

    @property
    def cur_iter(self):
        """Current iteration."""
        return len(self.solution_qualities) - 1

    @property
    def cur_index(self):
        """Current iteration."""
        return self.cur_iter % self.memory

    def prepare_loop(self):
        """Create storage arrays."""
        super().prepare_loop()

        self.dys = np.empty((self.memory, self.m), dtype=float)
        self.dss = np.empty((self.memory, self.m), dtype=float)
        self.dpriress = np.empty((self.memory, self.m), dtype=float)
        self.dduaress = np.empty((self.memory, self.m), dtype=float)

        # norms squared; we cache them to avoid recomputing self.memory times
        self.dydys = np.empty(self.memory, dtype=float)
        self.dsdss = np.empty(self.memory, dtype=float)
        self.dydss = np.empty(self.memory, dtype=float)
        self.dpriresdpriress = np.empty(self.memory, dtype=float)
        self.dduaresdduaress = np.empty(self.memory, dtype=float)

        self.old_y = np.empty(self.m, dtype=float)
        self.old_s = np.empty(self.m, dtype=float)
        self.old_prires = np.empty(self.m, dtype=float)
        self.old_duares = np.empty(self.m, dtype=float)

        self.pd_errors = []
        self.pd_errors_running_sum = 0.
        self.pd_scales = []
        self.regularizations = []

        self.used_memory = 0

    def iterate(self):
        """Simple Douglas Rachford iteration with Broyden update to override.
        """

        # compute primal-dual things, DR step is obtained from them
        self.s[:], self.y[:], self.pri_res[:], self.dua_res[:] = \
            self.compute_pridual_step(self.z)

        # compute norms once
        self.pri_res_norm = np.linalg.norm(self.pri_res)
        self.dua_res_norm = np.linalg.norm(self.dua_res)

        # update Broyden stores, update Broyden regularization
        if self.cur_iter > 0:

            self.dys[self.cur_index] = self.y - self.old_y
            self.dss[self.cur_index] = self.s - self.old_s
            self.dpriress[self.cur_index] = self.pri_res - self.old_prires
            self.dduaress[self.cur_index] = self.dua_res - self.old_duares

            self.dydys[self.cur_index] = self.dys[self.cur_index] @ self.dys[self.cur_index]
            self.dsdss[self.cur_index] = self.dss[self.cur_index] @ self.dss[self.cur_index]
            self.dydss[self.cur_index] = self.dys[self.cur_index] @ self.dss[self.cur_index]
            self.dpriresdpriress[self.cur_index] = self.dpriress[self.cur_index] @ self.dpriress[self.cur_index]
            self.dduaresdduaress[self.cur_index] = self.dduaress[self.cur_index] @ self.dduaress[self.cur_index]

            self.used_memory = min(self.used_memory + 1, self.memory)

            self.update_broyden_regularization()

        self.regularizations.append(self.broyden_regularizer)

        # we store the primal and dual res norms
        self.pd_errors.append(np.log(self.pri_res_norm / self.dua_res_norm))
        self.pd_errors_running_sum += self.pd_errors[-1]

        # we choose the scale
        self.update_pd_scale() # change z in place

        # and the primal dual scale chosen
        self.pd_scales.append(float(self.pd_scale))

        # Maybe: these 2 in a loop to accept/reject update changing regularization;
        # error prone, and not sure it is good idea; if we do be careful about
        # order of execution (where old's are saved)

        # compute primal and dual Broyden step
        pri_br_step, dua_br_step = self.compute_pridual_broyden_step()

        # update z
        self.z[:] = (self.y - dua_br_step) - self.pd_scale * (self.s + pri_br_step)

        # store old things
        self.old_y[:] = self.y
        self.old_s[:] = self.s
        self.old_prires[:] = self.pri_res
        self.old_duares[:] = self.dua_res
        self.old_prires_norm = float(self.pri_res_norm)
        self.old_duares_norm = float(self.dua_res_norm)

    # def callback_converged(self):
    #     """Override this to do diagnostics with the instance at convergence."""
    #     import matplotlib.pyplot as plt
    #     plt.semilogy(self.regularizations, label='regularizations')
    #     plt.show()
    #     plt.scatter(np.log(self.pd_scales[:-1]), np.diff(self.pd_errors))
    #     print(np.corrcoef(np.log(self.pd_scales[:-1]), np.diff(self.pd_errors)))
    #     plt.show()
    #     X = np.vstack([np.log(self.pd_scales[:-1]), np.log(self.regularizations)[:-1]]).T
    #     Y = np.diff(self.pd_errors)
    #     print(np.linalg.solve(X.T @ X, X.T @ Y))
    #     breakpoint()

    def serve_broyden_elements(self):
        """Serve pieces used for Broyden loop."""

        for back_index in range(self.used_memory):
            index = (self.cur_index - back_index) % self.memory
            yield (
                self.dss[index],
                self.dys[index],
                self.dpriress[index],
                self.dduaress[index],
                self.dsdss[index],
                self.dydys[index],
                self.dydss[index],
                self.dpriresdpriress[index],
                self.dduaresdduaress[index],
                )

    ###
    # Primal-Dual scaling
    ###

    pid_scale_corrector = 0.0 # doesn't seem to work

    def update_pd_scale(self):
        """PID update of PD scale."""
        error_t = self.pd_errors[-1]
        integral_error = self.pd_errors_running_sum
        derivative_error = self.pd_errors[-1] - self.pd_errors[-2] if len(self.pd_errors) > 1 else 0
        control = self.Kp * error_t + self.Ki * integral_error + self.Kd * derivative_error

        if self.internal_verbose:
            print("ITER", self.cur_iter, "PRIMAL RESIDUAL", self.pri_res_norm, "DUAL RESIDUAL", self.dua_res_norm)
            print("ITER", self.cur_iter, "ERROR", error_t, "INTEGRAL", integral_error, "DERIVATIVE", derivative_error, "CONTROL", control)

        new_scale = np.exp(control * (self.broyden_regularizer**self.pid_scale_corrector))

        if self.internal_verbose:
            print(f"ITER {self.cur_iter} CHANGING SCALE FROM {self.pd_scale} TO {new_scale}")
        self.pd_scale = new_scale

        # this is probably not needed
        self.z[:] = self.y - self.s * new_scale

    ###
    # Regularized Broyden logic
    ###

    def update_broyden_regularization(self):
        """Update Broyden regularization."""
        old_step_len = np.linalg.norm(self.pd_scale * self.old_prires + self.old_duares)
        cur_step_len = np.linalg.norm(self.pd_scale * self.pri_res + self.dua_res)

        if cur_step_len < old_step_len:
            if self.internal_verbose:
                print(f'ITER {self.cur_iter} CURRENT REGULARIZER {self.broyden_regularizer}, DECREASING')
            self.broyden_regularizer *= self.broyden_regularizer_decrement
        else:
            if self.internal_verbose:
                print(f'ITER {self.cur_iter} CURRENT REGULARIZER {self.broyden_regularizer}, INCREASING')
            self.broyden_regularizer *= self.broyden_regularizer_increment

        self.broyden_regularizer = np.clip(
            self.broyden_regularizer,
            self.broyden_regularizer_floor,
            self.broyden_regularizer_ceil)
        # breakpoint()

    def get_broyden_regularization_factor(self, norm_square_update):
        """Regularize the norm of a single Broyden update."""
        return np.sqrt(1. + norm_square_update * self.broyden_regularizer)

    def compute_pridual_broyden_step(self):
        """Base method to compute a Broyden-style approximate Newton step."""

        mystep_pri = np.array(self.pri_res)
        mystep_dua = np.array(self.dua_res)
        result_pri = np.zeros(self.m)
        result_dua = np.zeros(self.m)

        # this should be correct
        for (ds, dy, dprires, dduares, dsds, dydy, dyds,
            dpriresdprires, dduaresdduares) in self.serve_broyden_elements():

            dzdz = dydy + dsds * self.pd_scale**2 - 2 * self.pd_scale * dyds
            dstepdstep = dpriresdprires * self.pd_scale**2 + dduaresdduares

            # get regularizer
            reduction_factor = self.get_broyden_regularization_factor(
                dzdz/dstepdstep)

            new_dstep_component = (
                (self.pd_scale**2 * (mystep_pri @ dprires) + mystep_dua @ dduares) /
                dstepdstep)
            new_dstep_component_reduced = new_dstep_component / reduction_factor

            # careful with the signs, these are correct
            mystep_pri -= dprires * new_dstep_component_reduced
            mystep_dua -= dduares * new_dstep_component_reduced
            result_pri -= ds * new_dstep_component_reduced
            result_dua += dy * new_dstep_component_reduced

        # final correction
        result_pri -= mystep_pri
        result_dua -= mystep_dua

        return result_pri, result_dua


class BroydenEqMem10NNNCQR(BroydenEqNNNCQR):
    """With 10 memory."""
    memory = 20
    broyden_regularizer_floor = 1e-12
    internal_verbose = False
    use_numpy = True

# class BroydenEqMem20NNNCQR(BroydenEqNNNCQR):
#     """With 20 memory.

#     Removing cap_ceil makes portfolio_bad converge better, but that introduces
#     issue with simple LP, at very low residuals interaction with PD scaling,
#     oscillation (around 1e-10). Maybe we can make pd scale aware of current
#     Br regularization - lower regularization should make the system more
#     reactive to scale change. Also if we do "backtrack" loop it affects it.

#     This seems least bad, cap_ceil=1e5, converges on all (1k tests) but worst
#     on problem_one has oscillations and worst on PPB takes 80k iters.
#     """
#     internal_verbose = False
#     memory = 20
#     use_numpy = True

#     # need higher cap_ceil
#     # acceleration_cap = 5
#     # cap_decrease_factor = 0.9
#     # cap_increase_factor = 1.005
#     # cap_floor = 1.
#     cap_ceil = 1e5 # tried np.inf, see comment; 1e4 fails 1 instance of PPB;
