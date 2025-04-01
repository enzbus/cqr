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

import numpy as np
import scipy as sp

from .cones import (
    project_nonsymm_soc)

class TestCones(TestCase):
    """Unit tests for cones projections."""

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
