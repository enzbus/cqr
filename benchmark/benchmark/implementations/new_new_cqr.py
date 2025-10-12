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

class NewNewCQR(BaseSolver):
    """New idea for base CQR formulation."""

    max_iterations = 100000

    used_matrix = "matrix"
    used_b = "b"
    used_c = "c"
    use_numpy = True
    pd_scale = 1.0

    def change_scale(self, newscale):
        print(f"ITER {len(self.solution_qualities)} CHANGING SCALE FROM {self.pd_scale} TO {newscale}")
        y = self.cone_project(self.z)
        s = y - self.z
        s /= self.pd_scale
        self.pd_scale = newscale
        s *= self.pd_scale
        self.z[:] = y - s
        self.e = self.qr_matrix @ (
            self.pd_scale * self.qr_matrix.T @ getattr(self, self.used_b) - self.c_qr
            ) - self.pd_scale * getattr(self, self.used_b)

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

    def cone_project(self, z):
        """Project on y cone."""
        return self.composed_cone_project(
            z, has_zero=False, has_free=True, has_hsde=False)

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

class EquilibratedNewNewCQR(NewNewCQR):
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

class BroydenEqNNCQR(EquilibratedNewNewCQR):
    """Add basic Broyden logic."""

    memory = 50
    max_iterations = 100_000
    acceleration_cap = 100

    def prepare_loop(self):
        """Create storage arrays."""
        super().prepare_loop()
        self.dys = np.empty((self.memory, self.m), dtype=float)
        self.dss = np.empty((self.memory, self.m), dtype=float)
        # self.dzs_norms = np.empty(self.memory, dtype=float)
        self.dpriress = np.empty((self.memory, self.m), dtype=float)
        self.dduaress = np.empty((self.memory, self.m), dtype=float)
        # self.dsteps_norms = np.empty(self.memory, dtype=float)
        self.old_y = np.empty(self.m, dtype=float)
        self.old_s = np.empty(self.m, dtype=float)
        self.old_prires = np.empty(self.m, dtype=float)
        self.old_duares = np.empty(self.m, dtype=float)
        self.step = np.empty(self.m, dtype=float)
        self.pri_res = np.empty(self.m, dtype=float)
        self.dua_res = np.empty(self.m, dtype=float)
        self.used_memory = 0
        self.nonneg_activity = np.empty(self.nonneg, dtype=bool)
        self.old_nonneg_activity = np.empty(self.nonneg, dtype=bool)
        self.soc_activity = np.empty(len(self.soc), dtype=int)
        self.old_soc_activity = np.empty(len(self.soc), dtype=int)

    def compute_nonneg_activity(self, z):
        """Compute activity (bool) of nonneg cones."""
        return z[self.zero: self.zero+self.nonneg] > 0

    def compute_soc_activity(self, z):
        """Compute activity (-1, 0, 1) of soc cones."""
        result = np.empty(len(self.soc), dtype=int)
        cur = self.zero + self.nonneg
        for index, soc_size in enumerate(self.soc):
            z_cone = z[cur:cur+soc_size]
            t, x = z_cone[0], z_cone[1:]
            nrm = np.linalg.norm(x)
            if t > nrm:
                result[index] = 1
            elif t < -nrm:
                result[index] = -1
            else:
                result[index] = 0
            cur += soc_size
        assert cur == self.m
        return result

    def iterate(self):
        """Simple Douglas Rachford iteration with Broyden update to override.
        """
        if self.memory == 0: # fall back to non-broyden logic
            super().iterate()
            return

        cur_iter = len(self.solution_qualities)
        cur_index = cur_iter % self.memory

        # compute active set; will be factored in projection logic itself
        # will only need one storage each; probably soc works with just 1 bit
        self.nonneg_activity = self.compute_nonneg_activity(self.z)
        self.soc_activity = self.compute_soc_activity(self.z)
        if cur_iter > 0:
            if (np.all(self.nonneg_activity == self.old_nonneg_activity)
                    and np.all(self.soc_activity == self.soc_activity)):
                active_set_changed = False
            else:
                active_set_changed = True
            self.old_nonneg_activity[:] = self.nonneg_activity
            self.old_soc_activity[:] = self.soc_activity

        # compute DR step
        # self.y[:] = self.cone_project(self.z)
        # step_base = self.linspace_project_basic(2 * self.y - self.z) - self.y
        # self.step[:] = step_base + self.e
        self.s[:], self.y[:], self.pri_res[:], self.dua_res[:] = self.compute_pridual_step(self.z)

        # HERE THE LOGIC TO UPDATE THE SCALE
        self.update_pd_scale() # change z in place

        # update Broyden stores
        if cur_iter > 0:
            self.dys[cur_index] = self.y - self.old_y
            self.dss[cur_index] = self.s - self.old_s
            self.dpriress[cur_index] = self.pri_res - self.old_prires
            self.dduaress[cur_index] = self.dua_res - self.old_duares
            # self.dzs_norms[cur_index] = np.linalg.norm(self.dzs[cur_index])
        self.old_y[:] = self.y
        self.old_s[:] = self.s
        self.old_prires[:] = self.pri_res
        self.old_duares[:] = self.dua_res

        # # update dstep
        # if cur_iter > 0:
        #     self.dsteps[cur_index] = self.step - self.old_step
        #     # self.dsteps_norms[cur_index] = np.linalg.norm(
        #     #     self.dsteps[cur_index])
        # self.old_step[:] = self.step

        # update used_memory
        if cur_iter > 0:
            self.used_memory = min(self.used_memory + 1, self.memory)
        if active_set_changed: # we could have skipped saving them...
            print(f'ITER {cur_iter} SETTING USED_MEMORY TO ZERO B/C ACTIVITY CHANGE')
            self.used_memory = 0

        # update with Broyden step
        self.z[:] = self.z[:] - self.compute_broyden_step()

    def update_pd_scale(self):
        cur_iter = len(self.solution_qualities)
        print("ITER", cur_iter, "PRIMAL RESIDUAL", np.linalg.norm(self.pri_res), "DUAL RESIDUAL", np.linalg.norm(self.dua_res))
        # if np.abs(np.log10(np.linalg.norm(self.pri_res) / np.linalg.norm(self.dua_res)) > 3):
        #     breakpoint()

        # very simple logic, to start
        new_scale = (np.linalg.norm(self.pri_res) / np.linalg.norm(self.dua_res))**.25

        print(f"ITER {len(self.solution_qualities)} CHANGING SCALE FROM {self.pd_scale} TO {new_scale}")
        self.pd_scale = new_scale
        self.z[:] = self.y - self.s * new_scale
        # for b/w compatibility with dr_step old method
        self.e[:] = self.qr_matrix @ (
            self.pd_scale * self.qr_matrix.T @ getattr(self, self.used_b) - self.c_qr
            ) - self.pd_scale * getattr(self, self.used_b)

    def serve_broyden_elements(self):
        """Serve pieces used for Broyden loop."""

        cur_iter = len(self.solution_qualities)
        cur_index = cur_iter % self.memory

        for back_index in range(self.used_memory):
            index = (cur_index - back_index) % self.memory
            yield (
                self.dss[index],
                self.dys[index],
                self.dpriress[index],
                self.dduaress[index],
                )

    def compute_broyden_step(self):
        """Base method to compute a Broyden-style approximate Newton step."""
        mystep = self.pd_scale * self.pri_res + self.dua_res
        result = np.zeros_like(mystep)

        # this should be correct
        for _, (ds, dy, dprires, dduares) in enumerate(
            self.serve_broyden_elements()):

            dz = dy - self.pd_scale * ds
            dstep = self.pd_scale * dprires + dduares

            dz_norm = np.linalg.norm(dz)
            dstep_norm = np.linalg.norm(dstep)

            # correction by current index
            dstep_normed = dstep / dstep_norm
            dz_snormed = dz / dstep_norm
            acceleration = dz_norm / dstep_norm

            # we cap the acceleration
            if acceleration > self.acceleration_cap:
                reduction_factor = acceleration / self.acceleration_cap
            else:
                reduction_factor = 1.

            dstep_component_reduced = (mystep @ dstep_normed) / reduction_factor
            mystep -= dstep_normed * dstep_component_reduced
            result +=  (dz_snormed * dstep_component_reduced)

        # final correction
        result -= mystep

        return result
