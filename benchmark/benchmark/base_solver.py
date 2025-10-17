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

import logging

import numpy as np
import scipy as sp
# import tqdm

logger = logging.getLogger()
from .nonsymm_soc import project_nonsymm_soc

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
            logger.info(
                f'Program: m={self.m}, n={self.n}, nnz={self.matrix.nnz},'
                f' zero={self.zero}, nonneg={self.nonneg}, soc={self.soc}')

        self.x = np.zeros(self.n) if x0 is None else np.array(x0)
        assert len(self.x) == self.n
        self.y = np.zeros(self.m) if y0 is None else np.array(y0)
        assert len(self.y) == self.m

        # define Q to check solution quality
        self.hsde_q = self.build_hsde_q()

        # in these benchmarks we only deal with feasible programs
        self.status = "Optimal"

        self.prepare_loop()

        self.solution_qualities = []

        # Need to do work, self.x and self.y need to contain solution
        self.loop()

    ###
    # ELEMENTS TO POSSIBLY REDEFINE
    ###

    # NOTE! What you need to do:
    # Make sure self.obtain_x_and_y() returns the actual x and y you'd return
    # to the user (scaling, ...). Ideally use the self.callback_iterate
    # unchanged. If you can use the self.loop unchanged. Do whatever in the
    # self.prepare_loop and self.iterate. So, only relevant variables for BM
    # logic are self.x and self.y, we do not deal here with self.u, self.v,
    # or self.z. ALSO do not modify self.hsde_q, which we use to compute
    # solution quality.

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
            logger.info(
                f'Converged in {len(self.solution_qualities)} iterations!')
            raise StopIteration

    def loop(self):
        """Either use this default loop, or redefine based on your needs."""
        try:
            # for _ in tqdm.tqdm(range(self.max_iterations)):
            for _ in range(self.max_iterations):
                self.callback_iterate()
                self.iterate()
        except StopIteration:
            pass
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
        u = np.concatenate([self.x, self.y, [1.]])
        v = self.hsde_q @ u
        u_cone_proj = self.project_u(u)
        v_cone_proj = self.project_v(v)

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
        self, conic_variable, has_zero=False, has_free=False, has_hsde=False,
        has_hsde_first=False, nonsymm_soc=False):
        """Project on composed cone, allowing for alternative formulations.
        
        :param conic_variable: Variable to project
        :type conic_variable: np.array
        :param has_zero: First block is in zero cone
        :type has_zero: bool
        :param has_free: First block is in free cone
        :type has_free: bool
        :param has_hsde: Last element is HSDE variable
        :type has_hsde: bool
        :param nonsymm_soc: Use non-symmetric SOC projection.
        :type nonsymm_soc: bool

        :returns: Projection
        :rtype: np.array
        """

        result = np.empty_like(conic_variable)
        cur = 0

        # optional HSDE first part
        if has_hsde_first:
            result[0] = np.maximum(conic_variable[0], 0.)
            cur += 1

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
        for idx, soc_dim in enumerate(self.soc):
            if nonsymm_soc:
                result[cur:cur+soc_dim] = project_nonsymm_soc(
                    conic_variable[cur:cur+soc_dim], a=self.nonsymm_soc_a[idx])
            else:
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

    def multiply_jacobian_hsde_project(self, z, dz):
        """Multiply by Jacobian of projection on cone of HSDE variable z.

        :param z: Point at which the Jacobian is computed.
        :type z: np.array
        :param dz: Input array.
        :type dz: np.array

        :return: Multiplication of du by the Jacobian
        :rtype: np.array 
        """
        result = np.zeros_like(z)

        # x part + zero cone
        result[:self.n+self.zero] = dz[:self.n+self.zero]
        cur = self.n+self.zero

        # nonneg cone
        result[cur:cur+self.nonneg] = (
            z[cur:cur+self.nonneg] > 0.) * dz[cur:cur+self.nonneg]
        cur += self.nonneg

        # soc cones
        for soc_dim in self.soc:
            result[cur:cur+soc_dim] = \
                self.multiply_jacobian_second_order_project(
                    z[cur:cur+soc_dim], dz[cur:cur+soc_dim])
            cur += soc_dim
        assert cur == self.n + self.m

        # hsde variable
        result[-1] = (z[-1] > 0.) * dz[-1]

        return result

    @staticmethod
    def multiply_jacobian_second_order_project(z, dz):
        """Multiply by Jacobian of projection on second-order cone.

        We follow the derivation in `Solution Refinement at Regular Points of
        Conic Problems
        <https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf>`_.

        :param z: Point at which the Jacobian is computed.
        :type z: np.array
        :param dz: Input array.
        :type dz: np.array

        :return: Multiplication of dz by the Jacobian
        :rtype: np.array 
        """

        assert len(z) >= 2
        assert len(z) == len(dz)
        result = np.zeros_like(z)

        x, t = z[1:], z[0]

        norm_x = np.linalg.norm(x)

        if norm_x <= t:
            result[:] = dz
            return result

        if norm_x <= -t:
            return result

        dx, dt = dz[1:], dz[0]

        result[0] = norm_x * dt + x.T @ dx
        result[1:] = x * dt + (t + norm_x) * dx - t * x * (
            x.T @ dx) / (norm_x**2)
        return result / (2 * norm_x)

        # result[0] += dt / 2.
        # xtdx = x.T @ dx
        # result[0] += xtdx / (2. * norm_x)
        # result[1:] += x * ((dt - (xtdx/norm_x) * (t / norm_x))/(2 * norm_x))
        # result[1:] += dx * ((t + norm_x) / (2 * norm_x))
        # return result

    def _build_custom_q(self, mat, b , c):
        """Build HSDE Q matrix."""
        if hasattr(mat, 'todense'):
            mat = mat.todense()
        dense = np.block([
            [np.zeros((self.n, self.n)), mat.T , c.reshape(self.n, 1), ],
            [ -mat, np.zeros((self.m, self.m)), b.reshape(self.m, 1),],
            [-c.reshape(1, self.n), -b.reshape(1, self.m), np.zeros((1, 1)),],
        ])
        return sp.sparse.csc_array(dense)


if __name__ == "__main__":
    from tqdm import tqdm

    n = 20

    def _dense_grad(z):
        grad_cols = []
        for i in range(n): #tqdm(range(n)):
            e = np.zeros(n)
            e[i] = 1
            grad_cols.append(BaseSolver.multiply_jacobian_second_order_project(
                z, e))
        return np.array(grad_cols)

    def _func(z):
        result = np.zeros_like(z)
        BaseSolver.second_order_project(z, result)
        return result

    # plot eigenvals
    import matplotlib.pyplot as plt
    myz = np.random.randn(n)
    j = _dense_grad(myz)
    plt.plot(np.linalg.eigh(j)[0])
    plt.title('eivals of jacobian matrix of SOC project')
    plt.show()
    plt.plot(np.linalg.eigh(j@j)[0])
    plt.title('eivals of jacobian matrix squared of SOC project')
    plt.show()

    results = []
    for _ in range(100):
        np.random.seed(_)

        myz = np.random.randn(n)
        myu = _func(myz)
        myv = myu - myz
        j = _dense_grad(myz)
        # check few identities that should hold
        assert np.allclose(myu, j @ myu)
        assert np.allclose(0., j @ myv)

        result = sp.optimize.check_grad(
            _func, _dense_grad, x0 = myz,
            # epsilon=1e-12
            )
        results.append(result)
        # print("RESULT", result)

    print("MEAN", np.mean(results), "STD", np.std(results))

    # # simple test of cone projection derivative
    # m = 10
    # n = 30
    # zero = 0
    # nonneg = 0
    # soc = [10]#, 20, 30]
    # matrix = np.random.randn(m,n)
    # b = np.random.randn(m)
    # c = np.random.randn(n)
    # solver = BaseSolver(matrix, b, c, zero=zero, nonneg=nonneg, soc=soc)

    # def _dense_grad(z):
    #     grad_cols = []
    #     for i in tqdm(range(n+m+1)):
    #         e = np.zeros(n+m+1)
    #         e[i] = 1
    #         grad_cols.append(solver.multiply_jacobian_hsde_project(z, e))
    #     return np.array(grad_cols)

    # for _ in range(10):
    #     np.random.seed(_)
    #     print("RESULT", sp.optimize.check_grad(
    #         solver.project_u, _dense_grad, x0 = np.random.randn(n+m+1),
    #         epsilon=1e-12))
