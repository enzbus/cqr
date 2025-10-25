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
"""Branch off new_cqr module, to iron out a few more choices.

Minor code fixes here and there to simplify something.
"""

import numpy as np
import scipy as sp
from ..base_solver import BaseSolver
from pyspqr import qr

class TemporaryNewNewCQR(BaseSolver):
    """New idea for base CQR formulation."""

    max_iterations = 100000

    used_matrix = "matrix"
    used_b = "b"
    used_c = "c"
    use_numpy = True
    pd_scale = 1.0

    # def change_scale(self, newscale):
    #     print(f"ITER {len(self.solution_qualities)} CHANGING SCALE FROM {self.pd_scale} TO {newscale}")
    #     y = self.cone_project(self.z)
    #     s = y - self.z
    #     s /= self.pd_scale
    #     self.pd_scale = newscale
    #     s *= self.pd_scale
    #     self.z[:] = y - s
    #     self.e = self.qr_matrix @ (
    #         self.pd_scale * self.qr_matrix.T @ getattr(self, self.used_b) - self.c_qr
    #         ) - self.pd_scale * getattr(self, self.used_b)

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

        # shift in the linspace projector
        self.e = self.qr_matrix @ (
            self.pd_scale * self.qr_matrix.T @ getattr(self, self.used_b) - self.c_qr
            ) - self.pd_scale * getattr(self, self.used_b)

        self.z = np.zeros(self.m)
        self.y = np.zeros(self.m)
        self.s = np.zeros(self.m)
        self.x = np.zeros(self.n)

        self.nonneg_activity = np.empty(self.nonneg, dtype=bool)
        self.old_nonneg_activity = np.empty(self.nonneg, dtype=bool)
        self.soc_activity = np.empty(len(self.soc), dtype=int)
        self.old_soc_activity = np.empty(len(self.soc), dtype=int)

        self.pri_res_norms = []
        self.dua_res_norms = []
        self.pd_scales = []
        self.exponents = []

    def cone_project(self, z):
        """Project on y cone."""
        return self.composed_cone_project(
            z, has_zero=False, has_free=True, has_hsde=False)

    def linspace_project(self, y_plus_s):
        """Linspace project (y+s) -> y."""
        return self.nullspace @ (self.nullspace.T @ y_plus_s) + self.e

    def linspace_project_basic(self, y_plus_s):
        """Linspace project (y+s) -> y, w/out shift."""
        return y_plus_s - self.qr_matrix @ (self.qr_matrix.T @ y_plus_s)

    def compute_pridual_step(self, z):
        """Compute primal dual things, all descaled; steps are the residuals.
        
        So, in basic DR
        newz = y + dua_step + pd_scale * (-s + pri_step)
        
        """
        y = self.cone_project(z)
        s = y - z
        s /= self.pd_scale
        # step = dr_step(z)
        # pri_step = self.nullspace @ self.nullspace.T @ step
        # dua_step = self.qr_matrix @ self.qr_matrix.T @ step = step - pri_step
        # step = self.nullspace @ (self.nullspace.T @ y) + self.nullspace @ (self.nullspace.T @ s) - y - (self.nullspace @ self.nullspace.T @ getattr(
        #    self, self.used_b)) - self.qr_matrix @ self.c_qr
        # step = self.nullspace @ (self.nullspace.T @ (y + s - getattr(self, self.used_b))) - y - self.qr_matrix @ self.c_qr
        pri_step = self.nullspace @ (self.nullspace.T @ (s - getattr(self, self.used_b)))
        dua_step = -self.qr_matrix @ (self.qr_matrix.T @ y + self.c_qr)
        assert np.allclose(self.pd_scale * pri_step + dua_step, self.dr_step(z))
        return s, y, pri_step, dua_step

    def dr_step(self, z):
        """DR step."""
        y = self.cone_project(z)
        return self.linspace_project_basic(2 * y - z) - y + self.e

    def iterate(self):
        """Simple Douglas Rachford iteration."""
        self.y[:] = self.cone_project(self.z)
        step = self.linspace_project_basic(2 * self.y - self.z) - self.y + self.e
        # print(np.linalg.norm(step))
        self.z[:] = self.z + step
        breakpoint()

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        self.y[:] = self.cone_project(self.z)
        self.s[:] = (self.y - self.z) / self.pd_scale
        x_qr = self.qr_matrix.T @ (getattr(self, self.used_b) - self.s)
        if self.use_numpy:
            self.x[:] = sp.linalg.solve_triangular(self.triangular, x_qr, lower=False)
        else:
            self.x[:] = self.pyspqr_e.T @ sp.sparse.linalg.spsolve_triangular(
                self.pyspqr_r, x_qr, lower=False)

class TemporaryEquilibratedNewNewCQR(TemporaryNewNewCQR):
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
            print(r1, r2)
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


class BroydenEqNNNCQR(TemporaryEquilibratedNewNewCQR):
    """Redo base class; refactored Broyden2EqNNCqr with minor fixes.

    Pest test so far 2025-10-25; even if unfinished. Initial fixes already
    improve portfolio_bad, we're now safely well convergent on that
    one (1000 tests). It's slower b/c regularization is now more conservative.

    There was error in adacap before, minor, same logic but correct seems to
    improve. Also simplified it and it improves portfolio_bad, made it more
    stringent in fact. Refactored so that we can test other schemes. Pri-dual
    fix of Broyden update probably matters a lot. Also we're doing a lot of
    redundant computations now, need to streamline.
    """

    memory = 50
    max_iterations = 100_000

    # basic regularization scheme
    acceleration_cap = 5
    cap_decrease_factor = 0.9
    cap_increase_factor = 1.005
    cap_floor = 1.
    cap_ceil = 100.

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
        self.dys_norms = np.empty(self.memory, dtype=float)
        self.dss_norms = np.empty(self.memory, dtype=float)
        self.dpriress_norms = np.empty(self.memory, dtype=float)
        self.dduaress_norms = np.empty(self.memory, dtype=float)

        self.old_y = np.empty(self.m, dtype=float)
        self.old_s = np.empty(self.m, dtype=float)
        self.old_prires = np.empty(self.m, dtype=float)
        self.old_duares = np.empty(self.m, dtype=float)

        self.step = np.empty(self.m, dtype=float)
        self.pri_res = np.empty(self.m, dtype=float)
        self.dua_res = np.empty(self.m, dtype=float)

        self.pd_errors = []
        self.pd_scales = []

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
            self.dys_norms[self.cur_index] = np.linalg.norm(self.dys[self.cur_index])

            self.dss[self.cur_index] = self.s - self.old_s
            self.dss_norms[self.cur_index] = np.linalg.norm(self.dss[self.cur_index])

            self.dpriress[self.cur_index] = self.pri_res - self.old_prires
            self.dpriress_norms[self.cur_index] = np.linalg.norm(self.dpriress[self.cur_index])

            self.dduaress[self.cur_index] = self.dua_res - self.old_duares
            self.dduaress_norms[self.cur_index] = np.linalg.norm(self.dduaress[self.cur_index])

            self.used_memory = min(self.used_memory + 1, self.memory)

            self.update_broyden_regularization()

        # we store the primal and dual res norms
        self.pd_errors.append(np.log(self.pri_res_norm / self.dua_res_norm))

        # we choose the scale
        self.update_pd_scale() # change z in place

        # and the primal dual scale chosen
        self.pd_scales.append(float(self.pd_scale))

        # update with Broyden step
        br_step = self.compute_broyden_step()
        if np.all(np.abs(br_step) < 1e-16):
            breakpoint()
        if np.any(np.isnan(br_step)):
            breakpoint()

        self.z[:] = self.z[:] - br_step

        # store old things
        self.old_y[:] = self.y
        self.old_s[:] = self.s
        self.old_prires[:] = self.pri_res
        self.old_duares[:] = self.dua_res
        self.old_prires_norm = float(self.pri_res_norm)
        self.old_duares_norm = float(self.dua_res_norm)

    def serve_broyden_elements(self):
        """Serve pieces used for Broyden loop."""

        for back_index in range(self.used_memory):
            index = (self.cur_index - back_index) % self.memory
            yield (
                self.dss[index],
                self.dys[index],
                self.dpriress[index],
                self.dduaress[index],
                self.dss_norms[index],
                self.dys_norms[index],
                self.dpriress_norms[index],
                self.dduaress_norms[index],
                )

    ###
    # Primal-Dual scaling
    ###

    def update_pd_scale(self):
        """PID update of PD scale."""
        error_t = self.pd_errors[-1]
        integral_error = np.sum(self.pd_errors)
        derivative_error = self.pd_errors[-1] - self.pd_errors[-2] if len(self.pd_errors) > 1 else 0
        control = self.Kp * error_t + self.Ki * integral_error + self.Kd * derivative_error

        if self.internal_verbose:
            print("ITER", self.cur_iter, "PRIMAL RESIDUAL", self.pri_res_norm, "DUAL RESIDUAL", self.dua_res_norm)
            print("ITER", self.cur_iter, "ERROR", error_t, "INTEGRAL", integral_error, "DERIVATIVE", derivative_error, "CONTROL", control)

        new_scale = np.exp(control)

        if self.internal_verbose:
            print(f"ITER {self.cur_iter} CHANGING SCALE FROM {self.pd_scale} TO {new_scale}")
        self.pd_scale = new_scale
        self.z[:] = self.y - self.s * new_scale
        # for b/w compatibility with dr_step old method
        self.e[:] = self.qr_matrix @ (
            self.pd_scale * self.qr_matrix.T @ getattr(self, self.used_b) - self.c_qr
            ) - self.pd_scale * getattr(self, self.used_b)

    ###
    # Regularized Broyden logic
    ###

    def update_broyden_regularization(self):
        """Update Broyden regularization."""
        old_step_len = np.linalg.norm(self.pd_scale * self.old_prires + self.old_duares)
        cur_step_len = np.linalg.norm(self.pd_scale * self.pri_res + self.dua_res)

        if cur_step_len < old_step_len:
            if self.internal_verbose:
                print(f'ITER {self.cur_iter} CURRENT ACCEL CAP {self.acceleration_cap}, INCREASING')
            self.acceleration_cap *= self.cap_increase_factor
        else:
            if self.internal_verbose:
                print(f'ITER {self.cur_iter} CURRENT ACCEL CAP {self.acceleration_cap}, DECREASING')
            self.acceleration_cap *= self.cap_decrease_factor

        self.acceleration_cap = np.clip(self.acceleration_cap, self.cap_floor, self.cap_ceil)
        # breakpoint()

    def get_broyden_regularization_factor(self, acceleration):
        """Regularize the norm of a single Broyden update."""

        # we cap the acceleration
        if acceleration > self.acceleration_cap:
            reduction_factor = acceleration / self.acceleration_cap
        else:
            reduction_factor = 1.
        return reduction_factor

    def compute_broyden_step(self):
        """Base method to compute a Broyden-style approximate Newton step."""
        mystep = self.pd_scale * self.pri_res + self.dua_res
        result = np.zeros_like(mystep)

        # this should be correct
        for (
                ds, dy, dprires, dduares,
                ds_norm, dy_norm, dprires_norm, dduares_norm
            ) in self.serve_broyden_elements():

            dz = dy - self.pd_scale * ds
            dstep = self.pd_scale * dprires + dduares
            assert np.isclose(dprires @ dduares, 0.)

            dz_norm = np.linalg.norm(dz)
            dstep_norm = np.linalg.norm(dstep)

            # correction by current index
            dstep_normed = dstep / dstep_norm
            dz_snormed = dz / dstep_norm
            acceleration = dz_norm / dstep_norm

            # get regularizer
            reduction_factor = self.get_broyden_regularization_factor(acceleration)

            # apply the update
            dstep_component_reduced = (mystep @ dstep_normed) / reduction_factor
            mystep -= dstep_normed * dstep_component_reduced
            result +=  (dz_snormed * dstep_component_reduced)

        # final correction
        result -= mystep

        return result
