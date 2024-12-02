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
"""This module defines operations with sparse matrices.

Either compressed-sparse-rows or compressed-sparse-columns (one is the
transpose of the other) are supported.
"""

import numpy as np
import scipy.sparse as sp

from .config import NONUMBA

if not NONUMBA: # pragma: no cover
    import numba as nb

    from .config import CSCR_MATRIX, B, R, Z

if NONUMBA:

    def add_csc_matvec(csc_matrix, array, result, invert_sign):
            """Multiply CSC sparse matrix by vector, add value to result array.

            :param csc_matrix: Tuple representing a CSC sparse matrix: column
                pointers, row indices, and matrix entries.
            :type csc_matrix: tuple
            :param array: Input array.
            :type array: np.array
            :param result: Resulting array, will be overwritten.
            :type result: np.array
            :param invert_sign: Invert sign of result.
            :type invert_sign: bool
            """
            m = len(result)
            n = len(array)
            indptr, indices, data = csc_matrix
            csc_matrix = sp.csc_matrix((data, indices, indptr), shape=(m, n))
            if invert_sign:
                result -= csc_matrix @ array
            else:
                result += csc_matrix @ array

else: # pragma: no cover

    @nb.jit(nb.void(CSCR_MATRIX, R[:], R[:], B), nopython=True)
    def add_csc_matvec(
            csc_matrix, array, result, invert_sign):
        """Multiply CSC sparse matrix by vector, add value to result array.

        :param csc_matrix: Tuple representing a CSC sparse matrix: column
            pointers, row indices, and matrix entries.
        :type csc_matrix: tuple
        :param array: Input array.
        :type array: np.array
        :param result: Resulting array, will be overwritten.
        :type result: np.array
        :param invert_sign: Invert sign of result.
        :type invert_sign: bool
        """

        # same names as Scipy
        indptr, indices, data = csc_matrix
        n = len(indptr) - 1
        assert len(array) == n

        # need to set the result to 0. first
        # result[:] = 0.

        # iterate over columns
        for j in range(n):
            for k in range(indptr[j], indptr[j + 1]):
                if invert_sign:
                    result[indices[k]] -= data[k] * array[j]
                else:
                    result[indices[k]] += data[k] * array[j]

if NONUMBA:

    def add_csr_matvec(csr_matrix, array, result, invert_sign):
        """Multiply CSR sparse matrix by vector, add value to result array.

        :param csr_matrix: Tuple representing a CSR sparse matrix: row
            pointers, column indexes, and matrix entries.
        :type csr_matrix: tuple
        :param array: Input array.
        :type array: np.array
        :param result: Resulting array, will be overwritten.
        :type result: np.array
        :param invert_sign: Invert sign of result.
        :type invert_sign: bool
        """
        indptr, indices, data = csr_matrix
        m = len(result)
        n = len(array)
        indptr, indices, data = csr_matrix
        csr_matrix = sp.csr_matrix((data, indices, indptr), shape=(m, n))
        if invert_sign:
            result -= csr_matrix @ array
        else:
            result += csr_matrix @ array

else: # pragma: no cover

    @nb.jit(nb.void(CSCR_MATRIX, R[:], R[:], B), nopython=True)
    def add_csr_matvec(csr_matrix, array, result, invert_sign):
        """Multiply CSR sparse matrix by vector, add value to result array.

        :param csr_matrix: Tuple representing a CSR sparse matrix: row
            pointers, column indexes, and matrix entries.
        :type csr_matrix: tuple
        :param array: Input array.
        :type array: np.array
        :param result: Resulting array, will be overwritten.
        :type result: np.array
        :param invert_sign: Invert sign of result.
        :type invert_sign: bool
        """

        # same names as Scipy
        indptr, indices, data = csr_matrix

        m = len(indptr) - 1
        assert len(result) == m

        # need to set the result to 0. first
        # result[:] = 0.

        # iterate over rows
        for i in range(m):
            for k in range(indptr[i], indptr[i+1]):
                if invert_sign:
                    result[i] -= data[k] * array[indices[k]]
                else:
                    result[i] += data[k] * array[indices[k]]

def add_dense_matvec(matrix, array, result, invert_sign):
    """Multiply dense matrix by vector, adds value to result array.

    :param matrix: Dense matrix
    :type matrix: np.array
    :param array: Input array.
    :type array: np.array
    :param result: Resulting array, will be overwritten.
    :type result: np.array
    :param invert_sign: Invert sign of result.
    :type invert_sign: bool
    """

    if invert_sign:
        result[:] -= matrix @ array
    else:
        result[:] += matrix @ array

if not NONUMBA: # pragma: no cover
    add_dense_matvec = nb.jit(
        nb.void(R[:, ::1], R[::1], R[::1], B), nopython=True)(add_dense_matvec)


def add_dense_rmatvec(matrix, array, result, invert_sign):
    """Right multiply dense matrix by vector, adds value to result array.

    :param matrix: Dense matrix
    :type matrix: np.array
    :param array: Input array.
    :type array: np.array
    :param result: Resulting array, will be overwritten.
    :type result: np.array
    :param invert_sign: Invert sign of result.
    :type invert_sign: bool
    """

    if invert_sign:
        result[:] -= matrix.T @ array
    else:
        result[:] += matrix.T @ array

if not NONUMBA: # pragma: no cover
    add_dense_rmatvec = nb.jit(
        nb.void(R[:, ::1], R[::1], R[::1], B), nopython=True)(
            add_dense_rmatvec)

def add_hsde_matvec( # pylint: disable=too-many-arguments
        problem_matrix, b, c, array, result, invert_sign):
    """Multiply by matrix HSDE matrix, add value to result array.

    :param problem_matrix: Tuple representing the problem matrix A in CSC
        format.
    :type problem_matrix: tuple
    :param b: Right-hand side of the constraint system.
    :type b: np.array
    :param c: Cost vector.
    :type c: np.array
    :param array: Input array.
    :type array: np.array
    :param result: Resulting array, will be overwritten.
    :type result: np.array
    :param invert_sign: Invert sign of result.
    :type invert_sign: bool
    """
    n = len(c)
    m = len(b)
    assert len(result) == n+m+1
    assert len(array) == len(result)

    add_csr_matvec(
        problem_matrix, array[n:n+m], result[:n], invert_sign=invert_sign)

    add_csc_matvec(
        problem_matrix, array[:n], result[n:n+m], invert_sign=not invert_sign)

    if invert_sign:
        result[:n] -= c * array[-1]
        result[n:-1] -= b * array[-1]
        result[-1] += c @ array[:n]
        result[-1] += b.T@array[n:n+m]
    else:
        result[:n] += c * array[-1]
        result[n:-1] += b * array[-1]
        result[-1] -= c.T@array[:n]
        result[-1] -= b.T@array[n:n+m]

if not NONUMBA: # pragma: no cover
    add_hsde_matvec =  nb.jit( # explicit strides to mark inputs as contiguous
        nb.void(CSCR_MATRIX, R[::1], R[::1], R[::1], R[::1], B),
            nopython=True)(add_hsde_matvec)


def conjugate_gradient(
        matrix, array, result, r, p, w, max_iters, eps):
    r"""Approximately symmetric linear system by conjugate gradient method.

    We follow `the algorithm descrition at slide 22
    <https://web.stanford.edu/class/ee364b/lectures/conj_grad_slides.pdf>`_.

    :param matrix: Tuple representing the matrix in CSC format.
    :type matrix: tuple
    :param array: Right-hand side of the linear system.
    :type array: np.array
    :param result: Resulting array, will be overwritten. Its provided value
        is used as initial guess.
    :type result: np.array
    :param p: Array used for computation, will be overwritten.
    :type p: np.array
    :param r: Array used for computation, will be overwritten.
    :type r: np.array
    :param w: Array used for computation, will be overwritten.
    :type w: np.array
    :param max_iters: Maximum number of iterations.
    :type max_iters: int
    :param eps: Tolerance for convergence, :math:\|r\| <= \epsilon \|b\|`.
    :type eps: float

    :returns: Number of iterations used.
    :rtype: int
    """

    assert len(r) == len(p)
    assert len(p) == len(w)
    assert len(result) == len(r)
    assert len(array) == len(r)

    old_rho = 1.

    # for convergence check
    bnorm = np.linalg.norm(array)

    # print('eps * bnorm', eps * bnorm)

    # r0 = b - A@x
    r[:] = array
    add_csc_matvec(matrix, array=result, result=r, invert_sign=True)

    rho = r.T @ r

    for i in range(max_iters):

        # print('iter', i,'sqrt_rho', np.sqrt(rho))

        if np.sqrt(rho) < eps * bnorm:
            return i

        if i == 0:
            p[:] = r
        else:
            # p_{k+1} = r_{k+1} + (\rho_{k-1} / \rho_k-2) p_k
            p *= (rho / old_rho)
            p[:] += r

        # w = A @ p
        w[:] = 0
        add_csc_matvec(matrix, array=p, result=w, invert_sign=False)

        ptw = p.T @ w

        alpha = rho/ptw

        # x_{k+1} = x_k + \alpha_k p_k
        result[:] += alpha * p

        # r_{k+1} = r_k - \alpha_k A @ p_k
        r[:] -= alpha * w

        old_rho = rho
        rho = r.T @ r

    return max_iters # pragma: no cover

if not NONUMBA: # pragma: no cover
    conjugate_gradient = nb.jit( # strides to mark inputs as contiguous
        Z(CSCR_MATRIX, R[::1], R[::1], R[::1], R[::1], R[::1], Z, R),
            nopython=True)(conjugate_gradient)
