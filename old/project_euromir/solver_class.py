# Copyright 2024 Enzo Busseti
#
# This file is part of Project Euromir.
#
# Project Euromir is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Project Euromir is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Project Euromir. If not, see <https://www.gnu.org/licenses/>.
"""Solver class.

Idea:

Centralizes memory allocation, its managed memory translates to a struct in C.
Each method, which should be very small and simple, translates to a C function.
Experiments (new features, ...) should be done as subclasses.
"""

import numpy as np
import scipy as sp

from project_euromir import equilibrate


class Solver:
    """Solver class.

    :param matrix: Problem data matrix.
    :type n: sp.sparse.csc_matrix
    :param b: Dual cost vector.
    :type b: np.array
    :param c: Primal cost vector.
    :type c: np.array
    :param zero: Size of the zero cone.
    :type zero: int
    :param nonneg: Size of the non-negative cone.
    :type nonneg: int
    :param x0: Initial guess of the primal variable. Default None, equivalent
        to zero vector.
    :type x0: np.array or None.
    :param y0: Initial guess of the dual variable. Default None, equivalent
        to zero vector.
    :type y0: np.array or None.
    """

    def __init__(self, matrix, b, c, zero, nonneg, x0=None, y0=None):
        
        # process program data
        self.matrix = sp.csc_matrix(matrix)
        self.m, self.n = matrix.shape
        assert zero > 0
        assert nonneg > 0
        assert zero + nonneg == self.m
        self.zero = zero
        self.nonneg = nonneg
        assert len(b) == m
        self.b = np.array(b, dtype=float)
        assert len(c) == n
        self.c = np.array(c, dtype=float)
        self._equilibrate_program_data()

        # process initial guess
        self.x = np.empty(self.n, dtype=float)
        self.y = np.empty(self.m, dtype=float)
        self.update_variables(x0=x0, y0=y0)
        
    def update_variables(self, x0=None, y0=None):
        """Update initial values of the primal and dual variables.

        :param x0: Initial guess of the primal variable. Default None,
            equivalent to zero vector.
        :type x0: np.array or None.
        :param y0: Initial guess of the dual variable. Default None, equivalent
            to zero vector.
        :type y0: np.array or None.
        """

        if x0 is None:
            self.x[:] = np.zeros(self.n, dtype=float)
        else:
            assert len(x0) == self.n
            self.x[:] = np.array(x0, dtype=float)
        if y0 is None:
            self.y[:] = np.zeros(self.m, dtype=float)
        else:
            assert len(y0) == self.m
            self.y[:] = np.array(y0, dtype=float)
        self._equilibrate_transform_variables()

    def _equilibrate_program_data(self):
        """Apply Ruiz equilibration to the problem data."""
        self.d, self.e, self.sigma, self.rho, self.matrix_transf, \
            self.b_transf, self.c_transf = \
                equilibrate.hsde_ruiz_equilibration(
                    self.matrix, self.b, self.c, dimensions={
                        'zero': self.zero, 'nonneg': self.nonneg,
                        'second_order': ()},
                    max_iters=1000, eps_rows=1E-2, eps_cols=1E-2)

    def _equilibrate_transform_variables(self):
        """Transform the initial guess solution vectors with equilibration."""
        self.x_transf = (self.sigma * self.x) / self.e
        self.y_transf = (self.rho * self.y) / self.d

    def _qr_transform_program_data(self):
        """Apply QR decomposition to equilibrated program data."""
        assert m > n, "Case m <= n not yet implemented."

        q, r = np.linalg.qr(self.matrix_transf.todense(), mode='complete')
        self.matrix_qr_transf = q[:, :self.n].A
        self.nullspace_projector = q[:, self.n:].A
        self.r = r[:self.n]
        self.c_qr_transf = np.linalg.solve(self.r, self.c_transf)

        self.sigma_qr = np.linalg.norm(
            self.b_transf) #/ np.mean(np.linalg.norm(matrix_transf, axis=1))
        self.b_qr_transf = self.b_transf/self.sigma_qr

