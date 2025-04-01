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
"""Unit tests for the QL transform."""

from unittest import TestCase

import numpy as np
import scipy as sp

from .ql_transform import (
    data_ql_transform, forward_transform_ql, backward_transform_ql)

class TestQLTransform(TestCase):
    """Unit tests for the QL transform.

    Can subclass by overriding setUpClass for testing corner cases.
    """

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

        # create (u,v) in the subspace
        cls.u = np.random.randn(cls.m + cls.n + 1)
        cls.v = cls.Q_original @ cls.u

        # run transform
        cls.A_transf, cls.c_transf, cls.b_transf, cls.l = data_ql_transform(
            cls.A, cls.b, cls.c)
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

    def test_transform_consistent(self):
        """Check consistent."""
        scaler = sp.linalg.block_diag(np.linalg.inv(self.l), np.eye(self.m))
        self.assertAllClose(scaler.T @ self.Q_original @ scaler, self.Q_transf)

    def test_var_transform_inverts(self):
        """Test variables transform inverts."""

        tau_init, u1_init, u2_init = self.unpack_hsde_var(self.u)
        kappa_init, v1_init, v2_init = self.unpack_hsde_var(self.v)

        u1_transf, tau_transf, v1_transf, kappa_transf = forward_transform_ql(
            u1_init, tau_init, v1_init, kappa_init, self.n, self.l)
        u1_orig, tau_orig, v1_orig, kappa_orig = backward_transform_ql(
            u1_transf, tau_transf, v1_transf, kappa_transf, self.n, self.l)

        u_orig = self.pack_hsde_var(tau_orig, u1_orig, u2_init)
        v_orig = self.pack_hsde_var(kappa_orig, v1_orig, v2_init)

        self.assertAllClose(self.u, u_orig)
        self.assertAllClose(self.v, v_orig)

    def test_subspace_preserved(self):
        """Test subspace is preserved by transform."""

        self.assertAllClose(self.Q_original @ self.u, self.v)

        tau_init, u1_init, u2_init = self.unpack_hsde_var(self.u)
        kappa_init, v1_init, v2_init = self.unpack_hsde_var(self.v)

        u1_transf, tau_transf, v1_transf, kappa_transf = forward_transform_ql(
            u1_init, tau_init, v1_init, kappa_init, self.n, self.l)

        u_transf = self.pack_hsde_var(tau_transf, u1_transf, u2_init)
        v_transf = self.pack_hsde_var(kappa_transf, v1_transf, v2_init)

        self.assertAllClose(self.Q_transf @ u_transf, v_transf)


if __name__ == '__main__':
    from unittest import main
    main()
