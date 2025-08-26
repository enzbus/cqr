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
"""Original cone_prog_refine formulation (2018).
"""

import numpy as np
import scipy as sp

from cqr.equilibrate import hsde_ruiz_equilibration
from ..base_solver import BaseSolver

class SimpleCPR(BaseSolver):
    """Simple implementation of cone_prog_refine."""

    # class constants possibly overwritten by subclasses
    epsilon_convergence = 1e-12
    lsqr_iters = 10
    max_iterations = 100000//(lsqr_iters * 2)

    hsde_q_used = "hsde_q"


    def prepare_loop(self):
        """Define anything we need to re-use."""
        self.z = np.zeros(self.n + self.m + 1)
        self.z[-1] = 1.
        self.u = np.copy(self.z)
        self.matrix_solve = sp.sparse.linalg.splu(
            sp.sparse.eye(self.m + self.n + 1, format="csc")
            + getattr(self, self.hsde_q_used))

    def iterate(self):
        """Do one iteration."""
        self.u[:] = self.project_u(self.z)
        residual = self.u - getattr(self, self.hsde_q_used) @ self.u - self.z

        # works with the first part zeroed out also
        # step0 = self.matrix_solve.solve(2 * self.u - self.z) - self.u
        # step0[:self.n+self.zero] = 0.

        # doesn't work
        # step0 = self.u - self.z

        # trying this
        v = self.hsde_q @ self.u # jury's out on this
        # v = self.u - self.z # this makes no sense, gets always zero grad
        v_cone_proj = self.project_v(v)
        grad = self.hsde_q.T @ (v - v_cone_proj)
        step0 = -grad

        result = sp.sparse.linalg.lsqr(
            sp.sparse.linalg.LinearOperator(
                shape=(self.m+self.n+1, self.m+self.n+1),
                matvec=lambda dz: self.multiply_jacobian_dr(self.z, dz),
                rmatvec=lambda dz: self.multiply_jacobian_dr_transpose(
                    self.z, dz)), -residual,
                    # x0=...,
                    # damp=0.0,
                    x0=step0,
                    atol=0., btol=0., # without this we have random termination
                    iter_lim = self.lsqr_iters
                    )
        self.z[:] = self.z + result[0]
        # oldresidual = np.copy(residual)
        # oldresidual_norm = np.linalg.norm(oldresidual)
        # for i in range(100):
        #     newz = self.z + result[0]/(2**i)
        #     newu = self.project_u(newz)
        #     newresidual = newu - getattr(self, self.hsde_q_used) @ newu - newz
        #     if np.linalg.norm(newresidual) < oldresidual_norm:
        #         self.z[:] = newz
        #         break
        # else:
        #     print("Backtrack failed!")
        #     raise StopIteration("Backtrack failed")

    def multiply_jacobian_dr(self, z, dz):
        """Multiply by Jacobian of DR operator."""
        tmp = self.multiply_jacobian_hsde_project(z, dz)
        return tmp - getattr(self, self.hsde_q_used) @ tmp - dz

    def multiply_jacobian_dr_transpose(self, z, dz):
        """Multiply by Jacobian of DR operator transpose."""
        tmp = dz - getattr(self, self.hsde_q_used).T @ dz
        return self.multiply_jacobian_hsde_project(z, tmp) - dz

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        self.x = self.u[:self.n] / self.u[-1]
        self.y = self.u[self.n:-1] / self.u[-1]

class EquilibratedCPR(SimpleCPR):
    """With Ruiz equilibration."""

    hsde_q_used = "eq_hsde_q"

    def prepare_loop(self):
        """Define anything we need to re-use."""
        (self.equil_d, self.equil_e, self.equil_sigma, self.equil_rho,
            eqmatrix, eqb, eqc) =\
                hsde_ruiz_equilibration(
                    matrix=self.matrix, b=self.b, c=self.c,
                    l_norm=2.,
                    # l_norm=np.inf, # seems much worse than not equil
                    eps_rows=1E-2, eps_cols=1E-2, max_iters=10,
                    dimensions={
                            'zero': self.zero, 'nonneg': self.nonneg,
                            'second_order': self.soc})

        self.eq_hsde_q = self._build_custom_q(eqmatrix, eqb, eqc)

        super().prepare_loop()

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        x_equil = self.u[:self.n] / self.u[-1]
        y_equil = self.u[self.n:-1] / self.u[-1]
        self.x = (self.equil_e * x_equil) / self.equil_sigma
        self.y = (self.equil_d * y_equil) / self.equil_rho