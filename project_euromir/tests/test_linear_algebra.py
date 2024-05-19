# BSD 3-Clause License

# Copyright (c) 2024-, Enzo Busseti

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time
from unittest import TestCase, main

import numpy as np
import scipy.sparse as sp

import project_euromir as lib


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
