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
"""This module contains a Python implementation of Ruiz equilibration."""

# TODO: if we keep this, import test files from outer package

import logging

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

logger = logging.getLogger(__name__)


def _cones_separation_matrix(zero, nonneg, second_order):
    """Sparse matrix that maps entries into which cone they belong to."""
    return sp.block_diag(
        [sp.eye(zero+nonneg)] + [np.ones((1, el)) for el in second_order]
        + [1.])  # we add the one for use below


def hsde_ruiz_equilibration(  # pylint: disable=too-many-arguments
        matrix, b, c, dimensions, d=None, e=None, rho=1., sigma=1.,
        eps_rows=1E-1, eps_cols=1E-1, max_iters=25):
    """Ruiz equilibration of problem matrices for the HSDE system.

    :param matrix: Problem matrix.
    :type matrix: scipy.sparse.csc_matrix
    :param b: Right-hand side vector of the linear system.
    :type b: np.array
    :param b: Cost vector.
    :type b: np.array
    :param b: Dimensions of the problem cones.
    :type dimensions: dict
    :param d: Initial value of the row scaler; if None, the default, will be
        initialized with ones. If provided you must ensure that the entries
        corresponding to the same cone are all equal.
    :type d: np.array or None
    :param e: Initial value of the column scaler; if None, the default, will be
        initialized with ones.
    :type e: np.array or None
    :param rho: Initial value of the other scaler of c, default 1.
    :type rho: float
    :param sigma: Initial value of the other scaler of b, default 1.
    :type sigma: float
    :param eps_rows: Row scaling converges if the largest norm of a row is
        smaller than ``(1 + eps_rows)`` times the smallest.
    :type eps_rows: float
    :param eps_cols: Column scaling converges if the largest norm of a column
        is smaller than ``(1 + eps_cols)`` times the smallest.
    :type eps_cols: float
    :param max_iters: Maximum number of iterations.
    :type max_iters: int

    :returns: Diagonal equilibration vectors of rows and columns, rho, sigma,
        the equilibrated matrix, b and c. The equilibrator d of rows is
        guaranteed to have equal entries for rows corresponding to the same
        cone (unless you passed an initial guess which is not).
    :rtype: (np.array, np.array, float, float, scipy.sparse.csc_matrix,
        np.array, np.array)
    """

    cones_mapper = _cones_separation_matrix(**dimensions)
    cones_sizes = cones_mapper.sum(1).A1.ravel()

    m, n = matrix.shape

    d_and_rho = np.empty(m+1)
    e_and_sigma = np.empty(n+1)

    if d is None:
        d_and_rho[:-1] = 1.
    else:
        d_and_rho[:-1] = d
    if e is None:
        e_and_sigma[:-1] = 1.
    else:
        e_and_sigma[:-1] = e

    d_and_rho[-1] = rho
    e_and_sigma[-1] = sigma

    work_matrix = sp.diags(d_and_rho[:-1]
                           ) @ matrix @ sp.diags(e_and_sigma[:-1])
    work_b = e_and_sigma[-1] * sp.diags(d_and_rho[:-1]) @ b
    work_c = d_and_rho[-1] * sp.diags(e_and_sigma[:-1]) @ c

    norm_rows_and_c = np.empty(m+1)
    norm_cols_and_b = np.empty(n+1)

    for i in range(max_iters):

        norm_rows_and_c[:-1] = spl.norm(work_matrix, axis=1)**2
        norm_rows_and_c[:-1] += work_b**2
        norm_rows_and_c[:-1] = np.sqrt(norm_rows_and_c[:-1])
        norm_rows_and_c[-1] = np.linalg.norm(work_c)

        # here we apply the cones separation, each block gets equal values
        norm_rows_and_c = np.sqrt(
            cones_mapper.T @ ((cones_mapper @ norm_rows_and_c**2)/cones_sizes))

        norm_cols_and_b[:-1] = spl.norm(work_matrix, axis=0)**2
        norm_cols_and_b[:-1] += work_c**2
        norm_cols_and_b[:-1] = np.sqrt(norm_cols_and_b[:-1])
        norm_cols_and_b[-1] = np.linalg.norm(work_b)

        r1 = max(norm_rows_and_c[norm_rows_and_c > 0]
                 ) / min(norm_rows_and_c[norm_rows_and_c > 0])
        r2 = max(norm_cols_and_b[norm_cols_and_b > 0]
                 ) / min(norm_cols_and_b[norm_cols_and_b > 0])

        logger.info('Equilibration iter %s: r1=%s, r2=%s', i, r1, r2)
        if (r1-1 < eps_rows) and (r2-1 < eps_cols):
            logger.info('Equilibration converged.')
            break

        d_and_rho[norm_rows_and_c > 0] *= \
            norm_rows_and_c[norm_rows_and_c > 0]**(-0.5)
        e_and_sigma[norm_cols_and_b > 0] *= \
            ((m+1)/(n+1))**(0.25) * norm_cols_and_b[
                norm_cols_and_b > 0]**(-0.5)

        work_matrix = sp.diags(d_and_rho[:-1]
                               ) @ matrix @ sp.diags(e_and_sigma[:-1])
        work_b = e_and_sigma[-1] * sp.diags(d_and_rho[:-1]) @ b
        work_c = d_and_rho[-1] * sp.diags(e_and_sigma[:-1]) @ c

    else:
        logger.info('Equilibration reached max. number of iterations.')

    return (
        d_and_rho[:-1], e_and_sigma[:-1], e_and_sigma[-1], d_and_rho[-1],
        work_matrix, work_b, work_c)
