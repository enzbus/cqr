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

from ..base_solver import BaseSolver

class SimpleSCS(BaseSolver):
    """Simple implementation of original SCS (HSDE based)."""

    # class constants possibly overwritten by subclasses
    epsilon_convergence = 1e-12
    max_iterations = 100000

    def prepare_loop(self):
        """Define anything we need to re-use."""
        self.matrix_solve = sp.sparse.linalg.splu(
            sp.sparse.eye(self.m + self.n + 1, format="csc") + self.hsde_q)
        self.u = np.zeros(self.n + self.m + 1)
        self.u[-1] = 1.
        self.v = np.zeros(self.n + self.m + 1)

    def iterate(self):
        """Do one iteration"""
        u_tilde = self.matrix_solve.solve(self.u + self.v)
        self.u[:] = self.project_u(u_tilde - self.v)
        self.v[:] = self.v - u_tilde + self.u

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        self.x = self.u[:self.n] / self.u[-1]
        self.y = self.u[self.n:-1] / self.u[-1]
