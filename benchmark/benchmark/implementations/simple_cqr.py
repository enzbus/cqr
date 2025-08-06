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
"""Simple implementation of CQR prototype(s)."""

import numpy as np
import scipy as sp

from .simple_scs import SimpleSCS
from cqr.ql_transform import data_ql_transform


class SimpleCQR(SimpleSCS):
    """With "Abc" QR factorization."""

    hsde_q_used = "abc_cqr_q"

    # # to test model
    # def _build_custom_q_test_ordering(self, mat, b ,c):
    #     """Build HSDE Q matrix."""
    #     if hasattr(mat, 'todense'):
    #         mat = mat.todense()
    #     dense = np.block([
    #         [  np.zeros((self.m, self.m)), b.reshape(self.m, 1),-mat,],
    #         [-b.reshape(1, self.m), np.zeros((1, 1)), -c.reshape(1, self.n)],
    #         [ mat.T , c.reshape(self.n, 1), np.zeros((self.n, self.n)),],
    #     ])
    #     return sp.sparse.csc_array(dense)

    def prepare_loop(self):
        """Define anything we need to re-use."""

        A_transf, c_transf, b_transf, (q, scale), self.l = data_ql_transform(
            self.matrix.todense(), b=self.b, c=self.c)

        self.abc_cqr_q = self._build_custom_q(
            mat=A_transf, b=b_transf, c=c_transf)

        super().prepare_loop()

    def backward_transform_ql(
            self, ux, tau):
        """Transform solutions back onto original scaling.
        
        Simplified from implementation in current CQR module (only u, not v).
        """

        u_tmp = sp.linalg.solve_triangular(
            self.l, np.concatenate([[tau], ux]), lower=True)

        ux_orig, tau_orig = u_tmp[1:], u_tmp[0]

        return ux_orig, tau_orig

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        ux, tau = self.u[:self.n], self.u[-1]
        ux_orig, tau_orig = self.backward_transform_ql(ux, tau)
        self.x = ux_orig / tau_orig
        self.y = self.u[self.n:-1] / tau_orig
