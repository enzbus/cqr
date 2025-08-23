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
from ..base_solver import BaseSolver
from .simple_scs import DouglasRachfordSCS

class Broyden1SCS(DouglasRachfordSCS):
    """Just to test basic idea."""

    max_iterations = 100

    def prepare_loop(self):
        """Define anything we need to re-use."""
        super().prepare_loop()
        self.oldz = np.copy(self.z)
        self.oldstep = np.zeros_like(self.z)
        self.step = np.zeros_like(self.z)
        self.jaco_inv = -np.eye(self.m+self.n+1)

    def update_jaco_inv(self, dz, dstep):
        """Broyden update."""
        if len(self.solution_qualities) < 5:
            return
        if len(self.solution_qualities) % 5 == 0:
            self.jaco_inv = -np.eye(self.m+self.n+1)
        current = self.jaco_inv @ dstep
        dsn = dstep / (np.linalg.norm(dstep)**2)
        self.jaco_inv += np.outer(
            dz-current, dsn)
        # self.jaco_inv += np.outer(
        #   dz - self.jaco_inv @ dstep, dstep
        #   ) / (np.linalg.norm(dstep)**2)

        assert np.allclose(self.jaco_inv @ dstep, dz)
        # breakpoint()

    def iterate(self):
        """Do one iteration."""
        self.oldz[:] = self.z
        # self.z[:] = self.z - (-self.step)
        self.z[:] = self.z - self.jaco_inv @ self.step

        self.u[:] = self.project_u(self.z)
        self.oldstep[:] = self.step
        self.step[:] = self.matrix_solve.solve(
            2 * self.u - self.z) - self.u

        self.update_jaco_inv(
            dz = self.z - self.oldz,
            dstep = self.step - self.oldstep,
        )

class Broyden3SCS(DouglasRachfordSCS):
    """Same as B1, but with right update scheme."""

    max_iterations = 1000
    memory = 5

    def prepare_loop(self):
        """Define anything we need to re-use."""
        super().prepare_loop()
        self.oldz = np.copy(self.z)
        self.oldstep = np.zeros_like(self.z)
        self.alldz = np.zeros((self.memory, len(self.z)))
        self.alldstep = np.zeros((self.memory, len(self.z)))
        self.step = np.zeros_like(self.z)
        self.jaco_inv = -np.eye(self.m+self.n+1)
        self.last_index = 0

    def make_jaco_inv(self):
        if len(self.solution_qualities) < self.memory * 1.1:
            return
        self.jaco_inv = -np.eye(self.m+self.n+1)

        for i in range(1, self.memory + 1):
            used_index = (i + self.last_index) % self.memory
            # print("updating with used_index", used_index)
            dz = self.alldz[used_index]
            dstep = self.alldstep[used_index]
            current = self.jaco_inv @ dstep
            dsn = dstep / (np.linalg.norm(dstep)**2)
            self.jaco_inv += np.outer(
                dz-current, dsn)
            assert np.allclose(self.jaco_inv @ dstep, dz)

    def iterate(self):
        """Do one iteration."""
        self.oldz[:] = self.z
        # self.z[:] = self.z - (-self.step)
        self.z[:] = self.z - self.jaco_inv @ self.step

        self.u[:] = self.project_u(self.z)
        self.oldstep[:] = self.step
        self.step[:] = self.matrix_solve.solve(
            2 * self.u - self.z) - self.u

        self.last_index = len(
            self.solution_qualities) % self.memory
        self.alldz[self.last_index, :] = self.z - self.oldz
        self.alldstep[self.last_index, :] = self.step - self.oldstep
        # print("last index", self.last_index)

        self.make_jaco_inv()


class Broyden2SCS(Broyden1SCS):

    def prepare_loop(self):
        """Define anything we need to re-use."""
        super().prepare_loop()
        self.oldz = np.copy(self.z)
        self.oldstep = np.zeros_like(self.z)
        self.step = np.zeros_like(self.z)
        self.jaco = -np.eye(self.m+self.n+1)

    def update_jaco(self, dz, dstep):
        """Broyden update."""
        if len(self.solution_qualities) < 5:
            return
        if len(self.solution_qualities) % 5 == 0:
            self.jaco = -np.eye(self.m+self.n+1)
        current = self.jaco @ dz
        dzn = dz / (np.linalg.norm(dz)**2)
        self.jaco += np.outer(
            dstep-current, dzn)
        assert np.allclose(self.jaco @ dz, dstep)
        # breakpoint()

    def iterate(self):
        """Do one iteration."""
        self.oldz[:] = self.z
        # self.z[:] = self.z - (-self.step)
        self.z[:] = self.z - np.linalg.solve(
            self.jaco, self.step)

        self.u[:] = self.project_u(self.z)
        self.oldstep[:] = self.step
        self.step[:] = self.matrix_solve.solve(
            2 * self.u - self.z) - self.u

        self.update_jaco(
            dz = self.z - self.oldz,
            dstep = self.step - self.oldstep,
        )

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
                    self.z, dz)), -step,
                    x0=step,
                    damp=0.0, # doesn't seem to help
                    atol=0., btol=0., # without this we have random termination
                    iter_lim = 5)

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
