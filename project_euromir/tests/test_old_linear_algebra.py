# Copyright 2024 Enzo Busseti.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for sparse matrix operations, including equilibration."""

import time
from unittest import TestCase, main

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from solver.linear_algebra import (add_csc_matvec, add_csr_matvec,
                                   add_dense_matvec, add_dense_rmatvec,
                                   add_hsde_matvec, conjugate_gradient)


class TestLinearAlgebra(TestCase):
    """Unit tests for linear algebra operations."""

    def test_csc_matvec(self):
        """Test CSC matrix multiplication by vector."""
        timer_ourcode = 0.
        timer_scipy = 0.

        for i in range(10):

            np.random.seed(i)
            m = int(np.random.uniform(10, 100))
            n = int(np.random.uniform(10, 100))
            result = np.zeros(m)
            mat_scipy = sp.random(m, n, density=.2, format='csc')
            mat = (mat_scipy.indptr, mat_scipy.indices, mat_scipy.data)
            b = np.random.randn(n)

            s = time.time()
            add_csc_matvec(mat, b, result, False)
            timer_ourcode += time.time() - s

            s = time.time()
            scipy_result = mat_scipy @ b
            timer_scipy += time.time() - s

            self.assertTrue(np.allclose(result, scipy_result))

            result = np.zeros(m)
            add_csc_matvec(mat, b, result, True)

            self.assertTrue(np.allclose(result, -scipy_result))

        # print('our code:', timer_ourcode, 'scipy', timer_scipy)
        # self.assertLess(timer_ourcode, timer_scipy)

    def test_csr_matvec(self):
        """Test CSR matrix multiplication by vector."""

        for i in range(10):

            np.random.seed(i)
            m = int(np.random.uniform(10, 100))
            n = int(np.random.uniform(10, 100))
            result = np.zeros(m)
            mat_scipy = sp.random(m, n, density=.2, format='csr')
            mat = (mat_scipy.indptr, mat_scipy.indices, mat_scipy.data)
            b = np.random.randn(n)
            add_csr_matvec(mat, b, result, False)
            self.assertTrue(np.allclose(result, mat_scipy @ b))
            result = np.zeros(m)
            add_csr_matvec(mat, b, result, True)
            self.assertTrue(np.allclose(result, -mat_scipy @ b))

    def test_dense_matvec(self):
        """Test dense matrix multiplication by vector."""

        for i in range(10):

            np.random.seed(i)
            m = int(np.random.uniform(10, 100))
            n = int(np.random.uniform(10, 100))
            mat = np.random.randn(m, n)
            result = np.zeros(m)
            b = np.random.randn(n)
            add_dense_matvec(mat, b, result, False)
            self.assertTrue(np.allclose(result, mat @ b))
            result = np.zeros(m)
            b = np.random.randn(n)
            add_dense_matvec(mat, b, result, True)
            self.assertTrue(np.allclose(result, -mat @ b))

    def test_dense_rmatvec(self):
        """Test dense matrix right multiplication by vector."""

        for i in range(10):

            np.random.seed(i)
            m = int(np.random.uniform(10, 100))
            n = int(np.random.uniform(10, 100))
            mat = np.random.randn(m, n)
            result = np.zeros(n)
            b = np.random.randn(m)
            add_dense_rmatvec(mat, b, result, False)
            self.assertTrue(np.allclose(result, mat.T @ b))
            result = np.zeros(n)
            b = np.random.randn(m)
            add_dense_rmatvec(mat, b, result, True)
            self.assertTrue(np.allclose(result, -mat.T @ b))

    def test_hsde_matvec(self):
        """Test multiplication by HSDE matrix."""
        for (n, m) in [(3, 5), (1, 6), (30, 10), (3, 1)]:
            np.random.seed(n+m)
            mat_scipy = sp.random(m, n, density=.2, format='csc')
            mat = (mat_scipy.indptr, mat_scipy.indices, mat_scipy.data)
            b = np.random.randn(m)
            c = np.random.randn(n)
            u = np.random.randn(m+n+1)
            v = np.zeros(m+n+1)

            hsde = sp.bmat([
                [np.zeros((n, n)), mat_scipy.T, c.reshape(n, 1)],
                [-mat_scipy, np.zeros((m, m)), b.reshape(m, 1)],
                [-c.reshape(1, n), -b.reshape(1, m), 0]], format='csc')

            test_result = hsde @ u
            add_hsde_matvec(mat, b, c, u, v, False)
            self.assertTrue(np.allclose(test_result, v))
            v[:] = 0.
            add_hsde_matvec(mat, b, c, u, v, True)
            self.assertTrue(np.allclose(test_result, -v))

    def test_conjugate_gradient(self):
        """Test implementation of the CG algorithm."""
        n = 10
        max_iters = n * 10

        for seed in range(10):
            np.random.seed(seed)
            b = np.random.randn(n)
            matrix = np.random.randn(n, n)
            matrix = matrix.T @ matrix
            matrix = sp.csc_matrix(matrix)
            assert isinstance(matrix, sp.csc_matrix)
            assert np.all(matrix.todense().T == matrix.todense())
            x_scipy = spl.spsolve(matrix, b)
            scipy_error = np.linalg.norm(matrix @ x_scipy - b)
            # print('scipy error', scipy_error)
            result = np.zeros(n)
            r = np.empty(n)
            p = np.empty(n)
            w = np.empty(n)
            mat = (matrix.indptr, matrix.indices, matrix.data)
            iters = conjugate_gradient(
                mat, b, result, r, p, w, max_iters=max_iters, eps=1E-16)
            self.assertLess(iters, max_iters)
            cg_error = np.linalg.norm(matrix @ result - b)
            # print('CG error', cg_error)
            self.assertLess(cg_error, scipy_error * 2)

if __name__ == '__main__':
    main(warnings='error') # pragma: no cover
