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

import time  # on msw timing function calls <1s does not work
from unittest import TestCase

import numpy as np
import scipy.sparse as sp

import project_euromir as lib
from project_euromir.linear_algebra import Q_matvec

from .test_equilibrate import _make_Q


class TestLinearAlgebra(TestCase):

    # def test_import(self):
    #     """Test that the wheel installed correctly."""
    #     import project_euromir

    def test_csc(self):
        """Test CSC matvec."""

        m = 1000
        n = 1000
        tries = 100
        timers = np.empty(tries, dtype=float)

        for seed in range(tries):
            np.random.seed(seed)
            mat = sp.random(m=m, n=n, dtype=float, density=.01).tocsc()
            inp = np.random.randn(n)
            out = np.random.randn(m)
            out1 = np.array(out)
            mult = np.random.choice([-1., None, 1.])
            if mult is None:
                mult = np.random.randn()
            s = time.time()
            lib.add_csc_matvec(
                n=n, col_pointers=mat.indptr, row_indexes=mat.indices,
                mat_elements=mat.data, input=inp, output=out, mult=mult)
            timers[seed] = time.time() - s
            self.assertTrue(np.allclose(out, out1 + mult * (mat @ inp)))
        print(f'timer CSC {np.mean(timers):e}')

    def test_csc_degen(self):
        """Test csc matvec with degenerate matrices."""
        m = 20
        n = 30
        for mat in [
            sp.csc_matrix((m, n), dtype=float),  # empty matrix
            sp.hstack([  # first half cols are zero
                sp.csc_matrix((m, n//2), dtype=float),
                sp.csc_matrix(np.ones((m, n//2)))]),
            sp.hstack([  # second half cols are zero
                sp.csc_matrix(np.ones((m, n//2))),
                sp.csc_matrix((m, n//2), dtype=float)]),
            sp.vstack([  # first half rows are zero
                sp.csc_matrix((m//2, n), dtype=float),
                sp.csc_matrix(np.ones((m//2, n)))]),
            sp.vstack([  # second half rows are zero
                sp.csc_matrix(np.ones((m//2, n))),
                sp.csc_matrix((m//2, n), dtype=float)])
        ]:

            assert mat.shape == (m, n)
            assert isinstance(mat, sp.csc_matrix)

            inp = np.random.randn(n)
            out = np.random.randn(m)
            oldo = np.copy(out)
            mult = 3.
            lib.add_csc_matvec(
                n=n, col_pointers=mat.indptr, row_indexes=mat.indices,
                mat_elements=mat.data, input=inp, output=out, mult=mult)
            self.assertTrue(np.allclose(oldo + mult * (mat @ inp), out))

    def test_csr_degen(self):
        """Test csr matvec with degenerate matrices (copied from above)."""
        m = 20
        n = 30
        for mat in [
            sp.csc_matrix((m, n), dtype=float),  # empty matrix
            sp.hstack([  # first half cols are zero
                sp.csc_matrix((m, n//2), dtype=float),
                sp.csc_matrix(np.ones((m, n//2)))]),
            sp.hstack([  # second half cols are zero
                sp.csc_matrix(np.ones((m, n//2))),
                sp.csc_matrix((m, n//2), dtype=float)]),
            sp.vstack([  # first half rows are zero
                sp.csc_matrix((m//2, n), dtype=float),
                sp.csc_matrix(np.ones((m//2, n)))]),
            sp.vstack([  # second half rows are zero
                sp.csc_matrix(np.ones((m//2, n))),
                sp.csc_matrix((m//2, n), dtype=float)])
        ]:

            assert mat.shape == (m, n)
            mat = sp.csr_matrix(mat)

            inp = np.random.randn(n)
            out = np.random.randn(m)
            oldo = np.copy(out)
            mult = 3.
            lib.add_csr_matvec(
                m=m, row_pointers=mat.indptr, col_indexes=mat.indices,
                mat_elements=mat.data, input=inp, output=out, mult=mult)
            self.assertTrue(np.allclose(oldo + mult * (mat @ inp), out))

    def test_csr(self):
        """Test CSR matvec."""

        m = 1000
        n = 1000
        tries = 100
        timers = np.empty(tries, dtype=float)

        for seed in range(tries):
            np.random.seed(seed)
            mat = sp.random(m=m, n=n, dtype=float, density=.01).tocsr()
            inp = np.random.randn(n)
            out = np.random.randn(m)
            out1 = np.array(out)
            mult = np.random.choice([-1., None, 1.])
            if mult is None:
                mult = np.random.randn()

            s = time.time()
            lib.add_csr_matvec(
                m=m, row_pointers=mat.indptr, col_indexes=mat.indices,
                mat_elements=mat.data, input=inp, output=out, mult=mult)
            timers[seed] = time.time() - s
            self.assertTrue(np.allclose(out, out1 + mult * (mat @ inp)))

        print(f'timer CSR {np.mean(timers):e}')
