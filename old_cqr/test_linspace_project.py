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
"""Unit tests for the linear space projection."""

from unittest import TestCase

import cvxpy as cp
import numpy as np
import scipy as sp

from .linspace_project import linspace_project
from .ql_transform import data_ql_transform

class TestLinspaceProject(TestCase):
    """Unit tests for the linear space projection."""

    # TODO: factor these things into base test class
    def assertAllClose(self, *args, **kwargs):
        """Wrapper around np.allclose."""
        self.assertTrue(np.allclose(*args, **kwargs))

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.m = 300
        cls.n = 100
        cls.A = np.random.randn(cls.m, cls.n)
        x = np.random.randn(cls.n)
        z = np.random.randn(cls.m)
        y = np.maximum(z, 0.)
        s = y - z
        cls.c = -(cls.A.T @ y)
        cls.b = cls.A @ x + s
        cls.Q_original = cls.build_Q(cls.A, cls.b, cls.c)

        # run transform
        (cls.A_transf, cls.c_transf, cls.b_transf,
            (cls.orth_mat_transf, cls.scale), cls.l
            ) = data_ql_transform(cls.A, cls.b, cls.c)
        cls.Q_transf = cls.build_Q(cls.A_transf, cls.b_transf, cls.c_transf)

    @classmethod
    def unpack_hsde_var(cls, var):
        """Unpack HSDE variable."""
        assert len(var) == cls.m + cls.n + 1
        return var[0], var[1:1+cls.n], var[-cls.m:]

    @classmethod
    def pack_hsde_var(cls, var0, var1, var2):
        """Pack HSDE variable."""
        return np.concatenate([[var0], var1, var2])

    @classmethod
    def build_Q(cls, A, b, c):
        """Build Q matrix with new ordering."""
        return np.block([
            [np.zeros((1, 1)), -c.reshape(1, cls.n), -b.reshape(1, cls.m)],
            [c.reshape(cls.n, 1), np.zeros((cls.n, cls.n)), A.T],
            [ b.reshape(cls.m, 1), -A, np.zeros((cls.m, cls.m))],
        ])

    @staticmethod
    def project_with_cvxpy(u, v, Q):
        """Projection using CVXPY for testing."""
        u_star = cp.Variable(len(u))
        v_star = cp.Variable(len(v))
        objective = cp.Minimize(
            cp.sum_squares(u - u_star) + cp.sum_squares(v - v_star))
        constraints = [Q @ u_star == v_star]
        cp.Problem(objective, constraints).solve(solver='OSQP')
        return u_star.value, v_star.value

    def test_projection(self):
        """Test simple projection."""
        u = np.random.randn(self.n+self.m+1)
        v = np.random.randn(self.n+self.m+1)
        # v1 is identically 0.
        v[1:1+self.n] = 0.
        tau, u1, u2 = u[0], u[1:1+self.n], u[-self.m:]
        kappa, v2 = v[0], v[-self.m:]

        # CVXPY solution
        u_cp, v_cp = self.project_with_cvxpy(u, v, self.Q_transf)

        # solution
        (tau_star, u1_star, u2_star, kappa_star, v2_star
            ) = linspace_project(
                tau, u1, u2, kappa, v2, self.orth_mat_transf, self.scale)

        u_star = np.concatenate([[tau_star], u1_star, u2_star])
        kappa_v2_star = np.concatenate([[kappa_star], v2_star])

        self.assertAllClose(u_star, u_cp)
        self.assertAllClose(
            kappa_v2_star, np.concatenate([v_cp[:1], v_cp[-self.m:]]))


if __name__ == '__main__':
    from unittest import main
    main()
