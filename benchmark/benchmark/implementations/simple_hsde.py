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
"""Simple implementation of basic HSDE + DR.

Cost per iteration is roughly twice that of SimpleSCS (two Q matrix
factorization solves, two cone projections).
"""

import cvxpy as cp
import numpy as np
import scipy as sp

from ..base_solver import BaseSolver

class SimpleHSDECvxpy(BaseSolver):
    """Simple implementation of HSDE DR splitting - CVXPY projection."""

    # class constants possibly overwritten by subclasses
    epsilon_convergence = 1e-12
    max_iterations = 200000

    def base_prepare(self):
        """Base prepare loop."""
        self.u = np.zeros(self.n + self.m + 1)
        self.u[-1] = 1.
        self.v = np.zeros(self.n + self.m + 1)

        # careful! this is the DR variable
        self.u_iterate = np.copy(self.u)
        self.v_iterate = np.copy(self.v)

    def cvxpy_linspace_prepare(self):
        """Prepare loop for cvxpy linspace projection."""
        self.u_cvxpy_var = cp.Variable(self.m + self.n + 1)
        self.v_cvxpy_var = cp.Variable(self.m + self.n + 1)
        self.u0_cvxpy_par = cp.Parameter(self.m + self.n + 1)
        self.v0_cvxpy_par = cp.Parameter(self.m + self.n + 1)
        self.cvxpy_problem = cp.Problem(
            cp.Minimize(
                cp.sum_squares(self.u_cvxpy_var - self.u0_cvxpy_par)
                + cp.sum_squares(self.v_cvxpy_var - self.v0_cvxpy_par)),
            [self.hsde_q @ self.u_cvxpy_var == self.v_cvxpy_var]
        )

    def prepare_loop(self):
        """Define anything we need to re-use."""
        self.cvxpy_linspace_prepare()
        self.base_prepare()

    def linspace_project(self, u0, v0):
        """Project on Q @ u == v linspace."""
        self.u0_cvxpy_par.value = u0
        self.v0_cvxpy_par.value = v0
        self.cvxpy_problem.solve(
            solver='OSQP', verbose=False, ignore_dpp=False)
        return self.u_cvxpy_var.value, self.v_cvxpy_var.value

    def iterate(self):
        """Do one iteration"""
        self.u[:] = self.project_u(self.u_iterate)
        self.v[:] = self.project_v(self.v_iterate)
        u_linproj, v_linproj = self.linspace_project(
            2 * self.u - self.u_iterate, 2 * self.v - self.v_iterate)
        self.u_iterate[:] = self.u_iterate + u_linproj - self.u
        self.v_iterate[:] = self.v_iterate + v_linproj - self.v

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        self.x = self.u[:self.n] / self.u[-1]
        self.y = self.u[self.n:-1] / self.u[-1]


class SimpleHSDE(SimpleHSDECvxpy):
    """Simple implementation of HSDE DR splitting - NO CVXPY projection."""

    def linspace_prepare(self):
        """Prepare linspace projection."""
        mat = (
            sp.sparse.eye(self.m + self.n + 1, format="csc")
            + self.hsde_q.T @ self.hsde_q)
        self.matrix_solve = sp.sparse.linalg.splu(mat)

    def linspace_project(self, u0, v0):
        """Project on Q @ u == v linspace."""
        rhs = u0 + self.hsde_q.T @ v0
        u = self.matrix_solve.solve(rhs)
        return u, self.hsde_q @ u

    def prepare_loop(self):
        """Define anything we need to re-use."""
        self.linspace_prepare()
        self.base_prepare()
