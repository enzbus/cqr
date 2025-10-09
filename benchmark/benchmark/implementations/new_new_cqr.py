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
            self.qr_matrix.T @ getattr(self, self.used_b) - self.c_qr
            ) - getattr(self, self.used_b)

        self.z = np.zeros(self.m)
        self.y = np.zeros(self.m)
        self.s = np.zeros(self.m)
        self.x = np.zeros(self.n)

        # self.allsteps = []

    def cone_project(self, z):
        """Project on y cone."""
        return self.composed_cone_project(
            z, has_zero=False, has_free=True, has_hsde=False)

    def linspace_project_basic(self, y_plus_s):
        """Linspace project (y+s) -> y, w/out shift."""
        return y_plus_s - self.qr_matrix @ (self.qr_matrix.T @ y_plus_s)

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
        self.s[:] = self.y - self.z
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

    memory = 10
    max_iterations = 100_000
    do_accumulation = True
    # accumulation_threshold = 10000
    skip_threshold = 100

    def prepare_loop(self):
        """Create storage arrays."""
        super().prepare_loop()
        self.dzs = np.empty((self.memory, self.m), dtype=float)
        self.dzs_norms = np.empty(self.memory, dtype=float)
        self.dsteps = np.empty((self.memory, self.m), dtype=float)
        self.dsteps_norms = np.empty(self.memory, dtype=float)
        self.old_z = np.empty(self.m, dtype=float)
        self.old_step_base = np.empty(self.m, dtype=float)
        self.step = np.empty(self.m, dtype=float)
        self.cur_index = -1
        self.used_memory = self.memory

    def iterate(self):
        """Simple Douglas Rachford iteration with Broyden update to override.
        """
        if self.memory == 0:
            super().iterate()

        # self.cur_index = len(self.solution_qualities) % self.memory

        # self.dzs[self.cur_index] = self.z - self.old_z
        # self.dzs_norms[self.cur_index] = np.linalg.norm(self.dzs[self.cur_index])

        dz_candidate = self.z - self.old_z
        dznorm_candidate = np.linalg.norm(dz_candidate)
        self.old_z[:] = self.z

        self.y[:] = self.cone_project(self.z)
        step_base = self.linspace_project_basic(2 * self.y - self.z) - self.y
        self.step[:] = step_base + self.e

        # self.dsteps[self.cur_index] = step_base - self.old_step_base
        dstep_candidate = step_base - self.old_step_base
        # self.dsteps_norms[self.cur_index] = np.linalg.norm(self.dsteps[self.cur_index])
        dstepnorm_candidate = np.linalg.norm(dstep_candidate)
        self.old_step_base[:] = step_base

        # if self.do_accumulation and (
        #         len(self.solution_qualities) > self.memory + 1) and (
        #             dznorm_candidate / dstepnorm_candidate > self.accumulation_threshold):
        #     # if len(self.solution_qualities) > 5000:
        #     # breakpoint()
        #     # print(self.dzs_norms / self.dsteps_norms)
        #     # breakpoint()
        #     self.dzs[self.cur_index] += dz_candidate
        #     self.dzs_norms[self.cur_index] = np.linalg.norm(self.dzs[self.cur_index])
        #     self.dsteps[self.cur_index] += dstep_candidate
        #     self.dsteps_norms[self.cur_index] = np.linalg.norm(self.dsteps[self.cur_index])
        #     self.used_memory = max(self.used_memory - 1, 1)
            # print(self.dzs_norms / self.dsteps_norms)
            # and we don't update the index
            # breakpoint()
            # self.cur_index = (self.cur_index + 1) % self.memory
            # self.dzs[self.cur_index] = dz_candidate
            # self.dzs_norms[self.cur_index] = dznorm_candidate
            # self.dsteps[self.cur_index] = dstep_candidate
            # self.dsteps_norms[self.cur_index] = dstepnorm_candidate
            # self.used_memory = 1
        self.cur_index = (self.cur_index + 1) % self.memory
        self.dzs[self.cur_index] = dz_candidate
        self.dzs_norms[self.cur_index] = dznorm_candidate
        self.dsteps[self.cur_index] = dstep_candidate
        self.dsteps_norms[self.cur_index] = dstepnorm_candidate

        if (len(self.solution_qualities) > self.memory + 1) and (
                     dznorm_candidate / dstepnorm_candidate > self.skip_threshold):
            self.used_memory = 0
            if len(self.solution_qualities) > 5000:
                breakpoint()
        else:
            self.used_memory = min(self.memory + 1, self.memory)

        if len(self.solution_qualities) > self.memory + 1:
            newstep = self.compute_broyden_step()
            # if len(self.solution_qualities) > 50000:
            #     breakpoint()
            self.z[:] = self.z - newstep
        else:
            self.z[:] = self.z + self.step[:]

    def compute_broyden_step(self):
        """Base method to compute a Broyden-style approximate Newton step."""
        return -self.step[:]

    def serve_broyden_elements(self):
        """Serve pieces used for Broyden loop."""
        for back_index in range(self.used_memory):
            index = (self.cur_index - back_index) % self.memory
            yield (
                self.dzs[index],
                self.dzs_norms[index],
                self.dsteps[index],
                self.dsteps_norms[index]
                )

class SparseBasicBroydenEqNNCQR(BroydenEqNNCQR):
    """Full memory BrCQR, testing normalization."""
    max_iterations = 100_000
    acceleration_cap = 1000

    def compute_broyden_step(self):
        """N-Memory sparse update."""

        mystep = np.copy(self.step)
        result = np.zeros_like(mystep)

        # this should be correct
        for _, (dz, dz_norm, ds, ds_norm) in enumerate(
            self.serve_broyden_elements()):

            # correction by current index
            ds_normed = ds / ds_norm
            dz_snormed = dz / ds_norm
            acceleration = dz_norm / ds_norm

            # we cap the acceleration
            if acceleration > self.acceleration_cap:
                reduction_factor = acceleration / self.acceleration_cap
            else:
                reduction_factor = 1.

            ds_component_reduced = (mystep @ ds_normed) / reduction_factor
            mystep -= ds_normed * ds_component_reduced
            result +=  (dz_snormed * ds_component_reduced)
        # final correction
        result -= mystep

        return result

class SparseAccumulateBroydenEqNNCQR(BroydenEqNNCQR):
    """Testing accumulation of updates."""
    max_iterations = 100_000
    acceleration_cap = 20

    def compute_broyden_step(self):
        """N-Memory sparse update."""

        mystep = np.copy(self.step)
        result = np.zeros_like(mystep)

        # this should be correct
        for _, (dz, dz_norm, ds, ds_norm) in enumerate(
            self.serve_broyden_elements()):

            # correction by current index
            acceleration = dz_norm / ds_norm

            # we cap the acceleration
            if acceleration > self.acceleration_cap:
                reduction_factor = acceleration / self.acceleration_cap
            else:
                reduction_factor = 1.

            ds_normed = ds / ds_norm
            dz_snormed = dz / ds_norm
            ds_component_reduced = (mystep @ ds_normed) / reduction_factor
            mystep -= ds_normed * ds_component_reduced
            result +=  (dz_snormed * ds_component_reduced)
        # final correction
        result -= mystep

        return result
