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
"""Old idea to apply L-BFGS directly to solution quality metric.

It was very inefficient.
"""

import numpy as np
from scipy import optimize as opt

from ..base_solver import BaseSolver

class SimpleBFGS(BaseSolver):
    """L-BFGS applied directly to solution metric."""

    # class constants
    epsilon_convergence = 1e-8
    max_iterations = 1000000

    def _callback(self, current_u):
        """Changed logic from base."""
        self.u[:] = current_u
        self.obtain_x_and_y()
        self.solution_qualities.append(self.check_solution_quality())
        if self.solution_qualities[-1] < self.epsilon_convergence:
            # thankfully Scipy handles this
            raise StopIteration

    def _func_gradient(self, u):
        """Simple objective and gradient."""

        v = self.hsde_q @ u
        u_cone_proj = self.project_u(u)
        v_cone_proj = self.project_v(v)

        obj = 0.5 * (
            np.linalg.norm(u-u_cone_proj)**2
            + np.linalg.norm(v-v_cone_proj)**2)

        grad = u - u_cone_proj + self.hsde_q.T @ (v - v_cone_proj)

        return obj, grad

    def loop(self):
        """Either use this default loop, or redefine based on your needs."""
        result = opt.fmin_l_bfgs_b(
            func=self._func_gradient, x0=np.copy(self.u), approx_grad=False,
            callback=self._callback, maxiter=self.max_iterations, maxfun=1e10,
            factr=0.0, pgtol=0.0)
        result[2].pop('grad')
        print(
            'hsde var', result[0][-1], 'obj.', result[1], 'stats', result[2])
        self.u[:] = result[0]
        self.obtain_x_and_y()
        self.solution_qualities.append(self.check_solution_quality())

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        self.x = self.u[:self.n] / self.u[-1]
        self.y = self.u[self.n:-1] / self.u[-1]

    def prepare_loop(self):
        """Define anything we need to re-use."""
        self.u = np.zeros(self.n+self.m+1)
        self.u[-1] = 1.
