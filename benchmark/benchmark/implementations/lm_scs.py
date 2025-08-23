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
"""Test of SCS-DR using Levemberg-Marquardt.

To test; likely some issue with SOC derivative - need to do tests of derivative
operator(s) numerically.
"""

import numpy as np
import scipy as sp

from cqr.equilibrate import hsde_ruiz_equilibration
from .simple_scs import DouglasRachfordSCS
from .simple_cqr import SimpleCQR

class LevMarSCS(DouglasRachfordSCS):
    """SCS-DR using Levemberg-Marquardt."""

    # class constants possibly overwritten by subclasses
    epsilon_convergence = 1e-12
    max_iterations = 100000//10

    # # used in SCS algorithm
    # hsde_q_used = "hsde_q"

    def prepare_loop(self):
        """Define anything we need to re-use."""
        super().prepare_loop()
        self.tr_matrix_solve = sp.sparse.linalg.splu(
            sp.sparse.eye(self.m + self.n + 1, format="csc") + getattr(
                self, self.hsde_q_used).T)

    def iterate(self):
        """Do one iteration."""
        self.u[:] = self.project_u(self.z)
        step = self.matrix_solve.solve(2 * self.u - self.z) - self.u
        # print(np.linalg.norm(step))
        # breakpoint()

        result = sp.sparse.linalg.lsqr(
            sp.sparse.linalg.LinearOperator(
                shape=(self.m+self.n+1, self.m+self.n+1),
                matvec=lambda dz: self.multiply_jacobian_dr(self.z, dz),
                rmatvec=lambda dz: self.multiply_jacobian_dr_transpose(
                    self.z, dz)), -step, x0=step, atol=0., btol=0.,
                    iter_lim=5)

        self.z[:] = self.z + result[0]

    def multiply_jacobian_dr(self, z, dz):
        """Multiply by Jacobian of DR operator."""
        tmp = self.multiply_jacobian_hsde_project(z, dz)
        return self.matrix_solve.solve(2 * tmp - dz) - tmp

    def multiply_jacobian_dr_transpose(self, z, dz):
        """Multiply by Jacobian of DR operator transpose."""
        tmp = self.tr_matrix_solve.solve(dz)
        return self.multiply_jacobian_hsde_project(z, 2 * tmp - dz) - tmp

    def multiply_jacobian_hsde_project(self, z, dz):
        """Multiply by Jacobian of projection on cone of HSDE variable z.

        :param z: Point at which the Jacobian is computed.
        :type z: np.array
        :param dz: Input array.
        :type dz: np.array

        :return: Multiplication of du by the Jacobian
        :rtype: np.array 
        """
        result = np.zeros_like(z)

        # x part + zero cone
        result[:self.n+self.zero] = dz[:self.n+self.zero]
        cur = self.n+self.zero

        # nonneg cone
        result[cur:cur+self.nonneg] = (
            z[cur:cur+self.nonneg] > 0.) * dz[cur:cur+self.nonneg]
        cur += self.nonneg

        # soc cones
        for soc_dim in self.soc:
            result[cur:cur+soc_dim] = \
                self.multiply_jacobian_second_order_project(
                    z[cur:cur+soc_dim], dz[cur:cur+soc_dim])
            cur += soc_dim
        assert cur == self.n + self.m

        # hsde variable
        result[-1] = (z[-1] > 0.) * dz[-1]

        return result

    @staticmethod
    def multiply_jacobian_second_order_project(z, dz):
        """Multiply by Jacobian of projection on second-order cone.

        We follow the derivation in `Solution Refinement at Regular Points of
        Conic Problems
        <https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf>`_.

        :param z: Point at which the Jacobian is computed.
        :type z: np.array
        :param dz: Input array.
        :type dz: np.array

        :return: Multiplication of dz by the Jacobian
        :rtype: np.array 
        """

        assert len(z) >= 2
        assert len(z) == len(dz)
        result = np.zeros_like(z)

        x, t = z[1:], z[0]

        norm_x = np.linalg.norm(x)

        if norm_x <= t:
            result[:] = dz
            return result

        if norm_x <= -t:
            return result

        dx, dt = dz[1:], dz[0]

        result[0] += dt / 2.
        xtdx = x.T @ dx
        result[0] += xtdx / (2. * norm_x)
        result[1:] += x * ((dt - (xtdx/norm_x) * (t / norm_x))/(2 * norm_x))
        result[1:] += dx * ((t + norm_x) / (2 * norm_x))
        return result

class LevMarQRSCS(SimpleCQR, LevMarSCS):
    """Using QR transform."""


class LevMarRefSCS(LevMarSCS):
    """Using refinement model.
    
    It's just multiplying matrix and rhs by (I + Q).

    Sadly it seems much slower convergence.
    """

    def prepare_loop(self):
        """Define anything we need to re-use."""
        super().prepare_loop()

        # self.identity_minus_q = sp.sparse.eye(
        #    self.m + self.n + 1, format="csc") - self.hsde_q
        self.identity_plus_q = sp.sparse.eye(
            self.m + self.n + 1, format="csc") + getattr(
                self, self.hsde_q_used)

    def iterate(self):
        """Do one iteration."""
        self.u[:] = self.project_u(self.z)
        step = self.matrix_solve.solve(2 * self.u - self.z) - self.u
        step_times_iplusq = self.identity_plus_q @ step
        # print(np.linalg.norm(step))

        result = sp.sparse.linalg.lsqr(
            sp.sparse.linalg.LinearOperator(
                shape=(self.m+self.n+1, self.m+self.n+1),
                matvec=lambda dz: self.multiply_jacobian_dr(self.z, dz),
                rmatvec=lambda dz: self.multiply_jacobian_dr_transpose(
                    self.z, dz)), -step_times_iplusq, x0=step, atol=0.,
                    btol=0., iter_lim=5)

        self.z[:] = self.z + result[0]

    # def multiply_jacobian_dr(self, z, dz):
    #     """Multiply by Jacobian of DR operator."""
    #     tmp = self.multiply_jacobian_second_order_project_u(z, dz)
    #     return self.identity_minus_q @ tmp - dz

    # def multiply_jacobian_dr_transpose(self, z, dz):
    #     """Multiply by Jacobian of DR operator transpose."""
    #     tmp = self.identity_minus_q.T @ dz
    #     return self.multiply_jacobian_second_order_project_u(z, tmp) - dz

    def multiply_jacobian_dr(self, z, dz):
        """Multiply by Jacobian of DR operator."""
        return self.identity_plus_q @ super().multiply_jacobian_dr(z, dz)

    def multiply_jacobian_dr_transpose(self, z, dz):
        """Multiply by Jacobian of DR operator transpose."""
        return super().multiply_jacobian_dr_transpose(
            z, self.identity_plus_q.T @ dz)

class LevMarRefQRSCS(SimpleCQR, LevMarRefSCS):
    """Using QR transform."""
