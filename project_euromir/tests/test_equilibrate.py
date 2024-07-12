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

import logging
import time  # on msw timing function calls <1s does not work
from unittest import TestCase

import numpy as np
import scipy as sp

import project_euromir as lib
from project_euromir import equilibrate

# logging.basicConfig(level='INFO')


def _make_Q(matrix, b, c):
    m = len(b)
    n = len(c)
    return sp.sparse.bmat([
        [None, matrix.T, c.reshape(n, 1)],
        [-matrix, None, b.reshape(m, 1)],
        [-c.reshape(1, n), -b.reshape(1, m), None]
    ], format='csc')


class TestEquilibrate(TestCase):
    """Test equilibration functions."""

    def _spectrum_Q_more_stable(
            self, matrix, b, c, matrix_transf, b_transf, c_transf):
        """Check that the spectrum of Q after transf has fewer changes."""
        Q_orig = _make_Q(matrix, b, c)
        Q_transf = _make_Q(matrix_transf, b_transf, c_transf)

        spectrum_orig = np.linalg.eigh(Q_orig.toarray())[0]
        spectrum_transf = np.linalg.eigh(Q_transf.toarray())[0]

        diff_spectrum_orig = np.diff(spectrum_orig)
        diff_spectrum_transf = np.diff(spectrum_transf)

        # import matplotlib.pyplot as plt
        # plt.plot(spectrum_orig)
        # plt.plot(spectrum_transf)
        # plt.show()

        for thres in np.logspace(-8, 0, 9):
            print('thres', thres)
            self.assertLessEqual(
                np.sum(diff_spectrum_orig < thres),
                np.sum(diff_spectrum_transf < thres),
            )

    def _generate(self, m, n, density=.01, seed=0):
        """Generate problem data."""
        np.random.seed(seed)
        matrix = sp.sparse.random(
            m=m, n=n, format='csc', density=density,
            # this gives very bad unscaled matrices
            data_rvs=sp.stats.poisson(25, loc=10,).rvs)
        b = np.random.randn(m)
        c = np.random.randn(n)
        return matrix, b, c

    def test_base(self):
        """Test Python implementation."""
        m = 50
        n = 20

        for i, density in enumerate(np.linspace(0.01, 1, 10)):
            matrix, b, c = self._generate(m, n, density=density, seed=i)

            d, e, sigma, rho, matrix_transf, b_transf, c_transf = \
                equilibrate.hsde_ruiz_equilibration(
                    matrix, b, c, dimensions={
                        'zero': m//2, 'nonneg': m//2, 'second_order': ()})
            # breakpoint()

            print(np.linalg.norm(c_transf))
            print(np.linalg.norm(b_transf))
            print(np.linalg.norm(matrix_transf.toarray(), axis=1))
            print(np.linalg.norm(matrix_transf.toarray(), axis=0))

            self._spectrum_Q_more_stable(
                matrix, b, c, matrix_transf, b_transf, c_transf)

        # breakpoint()
