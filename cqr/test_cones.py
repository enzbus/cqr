# Copyright 2025 Enzo Busseti
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
"""Unit tests for cones projections."""

from unittest import TestCase, skip

import cvxpy as cp
import numpy as np
import scipy as sp

from .cones import project_nonsymm_soc

class TestCones(TestCase):
    """Unit tests for cones projections."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.n = 3
        # create lower triangular
        cls.l = np.random.randn(cls.n+1, cls.n+1)
        cls.l *= np.tri(*cls.l.shape)
        cls.l /= cls.l[0, 0]
        # make sure full rank
        assert np.min(np.linalg.svd(cls.l)[1]) > 0

    @staticmethod
    def cvxpy_project_primal_ql_cone(tau, u1, l):
        """Implementation of projection on primal QL cone using CVXPY.

        This is just used to check that cone is unchanged from transformation.
        """
        tau_u1_star = cp.Variable(len(u1) + 1)
        objective = cp.Minimize(
            cp.sum_squares(np.concatenate([[tau], u1]) - tau_u1_star))
        transformed_back = np.linalg.inv(l) @ tau_u1_star
        constraints = [transformed_back[0] >= 0]
        cp.Problem(objective, constraints).solve(solver='OSQP')
        return tau_u1_star.value[0], tau_u1_star.value[1:]

    def test_primal_ql_cone_project(self):
        """Test primal QL cone projection against CVXPY."""
        np.random.seed(0)

        # create variable
        tau = 1.
        u1 = np.random.randn(self.n)
        tau_star, u1_star = self.cvxpy_project_primal_ql_cone(tau, u1, self.l)
        self.assertTrue(np.isclose(tau, tau_star))
        self.assertTrue(np.allclose(u1, u1_star))

        tau = -1
        tau_star, u1_star = self.cvxpy_project_primal_ql_cone(tau, u1, self.l)
        self.assertTrue(np.isclose(0, tau_star))
        self.assertTrue(np.allclose(u1, u1_star))

    @staticmethod
    def cvxpy_project_dual_ql_cone(kappa, v1, l):
        """Implementation of projection on dual QL cone using CVXPY.

        This is just used to check that cone is unchanged from transformation.
        """
        kappa_v1_star = cp.Variable(len(v1) + 1)
        objective = cp.Minimize(
            cp.sum_squares(np.concatenate([[kappa], v1]) - kappa_v1_star))
        transformed_back = l.T @ kappa_v1_star
        constraints = [transformed_back[0] >= 0, transformed_back[1:] == 0.]
        cp.Problem(objective, constraints).solve(solver='OSQP')
        return kappa_v1_star.value[0], kappa_v1_star.value[1:]

    def test_dual_ql_cone_project(self):
        """Test dual QL cone projection against CVXPY."""

        # create variable
        kappa = 1.
        v1 = np.random.randn(self.n)
        kappa_star, v1_star = self.cvxpy_project_dual_ql_cone(
            kappa, v1, self.l)
        self.assertTrue(np.isclose(kappa, kappa_star))
        self.assertTrue(np.allclose(v1_star, 0.))

        kappa = -1.
        kappa_star, v1_star = self.cvxpy_project_dual_ql_cone(
            kappa, v1, self.l)
        self.assertTrue(np.isclose(kappa_star, 0.))
        self.assertTrue(np.allclose(v1_star, 0.))

    # @skip(reason="We're not using this cone.")
    def test_nonsymm_soc(self):
        """Test projection on non-symmetric SOC."""
        np.random.seed(0)
        N = 100
        NTRIES = 1000
        for _ in range(NTRIES):
            x = np.random.randn(N)
            # chosen so that 3 clauses are more or less equally likely
            a = np.random.uniform(0, 1e-1 if _ % 2 == 0 else 1000., N-1)
            pi = project_nonsymm_soc(x, a)

            ACCURACY = 1e-12 # about the max we achieve with this prototype

            # check pi in cone
            self.assertGreaterEqual(
                pi[0] - np.linalg.norm(pi[1:] * a), -ACCURACY)

            # check pi - x in dual cone
            diff = pi - x
            self.assertGreaterEqual(
                diff[0] - np.linalg.norm(diff[1:] / a), -ACCURACY)

            # check pi orthogonal to pi - x
            self.assertLess(abs(np.dot(pi, diff)), ACCURACY)


if __name__ == '__main__':
    from unittest import main
    main()
