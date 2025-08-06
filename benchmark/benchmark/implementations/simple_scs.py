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
"""Simple implementation of original SCS algorithm (O'Donoghue et al., '16)."""

import numpy as np
import scipy as sp

from cqr.equilibrate import hsde_ruiz_equilibration
from ..base_solver import BaseSolver

class SimpleSCS(BaseSolver):
    """Simple implementation of original SCS (HSDE based)."""

    # class constants possibly overwritten by subclasses
    epsilon_convergence = 1e-12
    max_iterations = 100000

    # used in SCS algorithm
    hsde_q_used = "hsde_q"

    # used in subclasses
    def _build_custom_q(self, mat, b , c):
        """Build HSDE Q matrix."""
        if hasattr(mat, 'todense'):
            mat = mat.todense()
        dense = np.block([
            [np.zeros((self.n, self.n)), mat.T , c.reshape(self.n, 1), ],
            [ -mat, np.zeros((self.m, self.m)), b.reshape(self.m, 1),],
            [-c.reshape(1, self.n), -b.reshape(1, self.m), np.zeros((1, 1)),],
        ])
        return sp.sparse.csc_array(dense)

    def prepare_loop(self):
        """Define anything we need to re-use."""
        self.matrix_solve = sp.sparse.linalg.splu(
            sp.sparse.eye(self.m + self.n + 1, format="csc")
            + getattr(self, self.hsde_q_used))
        self.u = np.zeros(self.n + self.m + 1)
        self.u[-1] = 1.
        self.v = np.zeros(self.n + self.m + 1)

    def iterate(self):
        """Do one iteration"""
        u_tilde = self.matrix_solve.solve(self.u + self.v)
        # print(np.linalg.norm(u_tilde - self.u))
        self.u[:] = self.project_u(u_tilde - self.v)
        self.v[:] = self.v - u_tilde + self.u

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        self.x = self.u[:self.n] / self.u[-1]
        self.y = self.u[self.n:-1] / self.u[-1]


class EquilibratedSCS(SimpleSCS):
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
                    eps_rows=1E-1, eps_cols=1E-1, max_iters=25,
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
