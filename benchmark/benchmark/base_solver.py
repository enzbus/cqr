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
"""Base solver with common utilities, projections, benchmarking metrics."""

# pylint: disable=too-many-instance-attributes,too-many-positional-arguments,
# pylint: disable=too-many-arguments

import numpy as np
import scipy as sp


class BaseSolver:
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
    :param x0: Initial guess of the primal variable. Default None,
        equivalent to zero vector.
    :type x0: np.array or None.
    :param y0: Initial guess of the dual variable. Default None,
        equivalent to zero vector.
    :type y0: np.array or None.
    """

    def __init__(
            self, matrix, b, c, zero, nonneg, soc=(), x0=None, y0=None,
            verbose=True):

        # process program data
        self.matrix = sp.sparse.csc_matrix(matrix)
        self.m, self.n = matrix.shape
        assert zero >= 0
        assert nonneg >= 0
        for soc_dim in soc:
            assert soc_dim > 1
        assert zero + nonneg + sum(soc) == self.m
        self.zero = zero
        self.nonneg = nonneg
        self.soc = soc
        assert len(b) == self.m
        self.b = np.array(b, dtype=float)
        assert len(c) == self.n
        self.c = np.array(c, dtype=float)
        self.verbose = verbose

        if self.verbose:
            print(
                f'Program: m={self.m}, n={self.n}, nnz={self.matrix.nnz},'
                f' zero={self.zero}, nonneg={self.nonneg}, soc={self.soc}')

        self.x = np.zeros(self.n) if x0 is None else np.array(x0)
        assert len(self.x) == self.n
        self.y = np.zeros(self.m) if y0 is None else np.array(y0)
        assert len(self.y) == self.m

        # define Q to check solution quality
        self.hsde_q = self.build_hsde_q()

        # in these benchmarks we only deal with feasible programs
        self.status = "optimal"

        self.prepare_loop()

        self.solution_qualities = []

        # Need to do work, self.x and self.y need to contain solution
        self.loop()

    ###
    # ELEMENTS TO POSSIBLY REDEFINE
    ###

    # class constants possibly overwritten by subclasses
    epsilon_convergence = 1e-12
    max_iterations = 100000

    def callback_iterate(self):
        """You can probably re-use this with custom loops.

        :raises StopIteration:
        """
        self.obtain_x_and_y()
        self.solution_qualities.append(self.check_solution_quality())
        if self.solution_qualities[-1] < self.epsilon_convergence:
            print(f'Converged in {len(self.solution_qualities)} iterations!')
            raise StopIteration

    def loop(self):
        """Either use this default loop, or redefine based on your needs."""
        self.prepare_loop()
        for _ in range(self.max_iterations):
            self.callback_iterate()
            self.iterate()
        self.obtain_x_and_y()

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        # self.x = ...
        # self.y = ...

    def prepare_loop(self):
        """Define anything we need to re-use."""

    def iterate(self):
        """Do one iteration"""

    ###
    # UTILITY LOGIC FOR BENCHMARKING
    ###

    def project_u(self, u):
        """Utility projection of HSDE variable u.

        :param u: First variable of HSDE.
        :type u: np.array

        :return: Projection of u on its HSDE cone.
        :rtype: np.array 
        """
        return np.concatenate(
            [u[:self.n], self.composed_cone_project(
                conic_variable=u[self.n:], has_free=True, has_hsde=True)])

    def project_v(self, v):
        """Utility projection of HSDE variable v.

        :param v: Second variable of HSDE.
        :type v: np.array

        :return: Projection of v on its HSDE cone.
        :rtype: np.array 
        """
        return np.concatenate(
            [np.zeros(self.n), self.composed_cone_project(
                conic_variable=v[self.n:], has_zero=True, has_hsde=True)])

    def check_solution_quality(self):
        """Check quality of current x and y.

        :returns: Solution quality.
        :rtype: float
        """
        u = np.concatenate([self.x, self.y, 1.])
        v = self.hsde_q @ u
        u_cone_proj = self.project_u(u)
        v_cone_proj = self.project_v(u)

        # very basic metric of solution quality, relies on each test program
        # from same class being similarly scaled
        return np.sqrt(
            np.linalg.norm(u - u_cone_proj)**2
            + np.linalg.norm(v - v_cone_proj)**2)

    def build_hsde_q(self):
        """Build HSDE Q matrix.
        
        :returns: Q matrix
        :rtype: sp.sparse.csc_array
        """
        mat = self.matrix.todense()
        dense = np.block([
            [np.zeros((self.n, self.n)), mat.T , self.c.reshape(self.n, 1), ],
            [ -mat, np.zeros((self.m, self.m)), self.b.reshape(self.m, 1),],
            [-self.c.reshape(1, self.n), -self.b.reshape(1, self.m),
                np.zeros((1, 1)),],
        ])
        return sp.sparse.csc_array(dense)

    ###
    # CONE PROJECTION LOGIC, SHOULD BE FLEXIBLE ENOUGH
    ###

    # see options for compatibility across models
    def composed_cone_project(
        self, conic_variable, has_zero=False, has_free=False, has_hsde=False):
        """Project on composed cone, allowing for alternative formulations.
        
        :param conic_variable: Variable to project
        :type conic_variable: np.array
        :param has_zero: First block is in zero cone
        :type has_zero: bool
        :param has_free: First block is in free cone
        :type has_free: bool
        :param has_hsde: Last element is HSDE variable
        :type has_hsde: bool

        :returns: Projection
        :rtype: np.array
        """

        result = np.empty_like(conic_variable)
        cur = 0

        # optional zero or free cone part
        if has_zero:
            result[:self.zero] = 0.
            cur += self.zero
        if has_free:
            assert not has_zero
            result[:self.zero] = conic_variable[:self.zero]
            cur += self.zero

        # nonneg part always there
        result[cur:cur+self.nonneg] = np.maximum(
            conic_variable[cur:cur+self.nonneg], 0.)
        cur += self.nonneg

        # SOC part always there
        for soc_dim in self.soc:
            self.second_order_project(
                conic_variable[cur:cur+soc_dim], result[cur:cur+soc_dim])
            cur += soc_dim

        # optional HSDE part
        if has_hsde:
            result[-1] = np.maximum(conic_variable[-1], 0.)
            cur += 1

        assert cur == len(conic_variable)

        return result

    @staticmethod
    def second_order_project(z, result):
        """Project on second-order cone.

        :param z: Input array.
        :type z: np.array
        :param result: Resulting array.
        :type result: np.array
        """

        assert len(z) >= 2

        y, t = z[1:], z[0]

        # cache this?
        norm_y = np.linalg.norm(y)

        if norm_y <= t:
            result[:] = z
            return

        if norm_y <= -t:
            result[:] = 0.
            return

        result[0] = 1.
        result[1:] = y / norm_y
        result *= (norm_y + t) / 2.
