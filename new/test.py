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
"""Unit tests of the solver class."""

import time
import warnings
from unittest import TestCase, main

import cvxpy as cp
import numpy as np
import scipy as sp

from solver import Solver, Infeasible, Unbounded

class TestSolverClass(TestCase):
    """Unit tests of the solver class."""

    @staticmethod
    def make_program_from_matrix(matrix, seed=0):
        """Make simple LP program."""
        m,n = matrix.shape
        np.random.seed(seed)
        z = np.random.randn(m)
        y = np.maximum(z, 0.)
        s = y - z
        x = np.random.randn(n)
        b = matrix @ x + s
        c = -matrix.T @ y
        return b, c

    @staticmethod
    def make_program_from_cvxpy(problem_obj):
        """Make program from cvxpy Problem object."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            data = problem_obj.get_problem_data('ECOS')[0]
        assert data['dims'].zero == 0 # will need to update base methods
        # the eq components are data['A'] and data['b']
        return data['G'], data['h'], data['c']

    def check_solution_valid(self, matrix, b, c, x, y):
        """Check a LP solution is valid."""
        self.assertGreater(np.min(y), -1e-6)
        s = b - matrix @ x
        self.assertGreater(np.min(s), -1e-6)
        self.assertTrue(np.isclose(c.T @ x + b.T @ y, 0., atol=1e-6, rtol=1e-6))
        self.assertTrue(np.allclose(c, - matrix.T @ y, atol=1e-6, rtol=1e-6))

    @staticmethod
    def solve_program_cvxpy(A, b, c):
        """Solve simple LP with CVXPY."""
        m, n = A.shape
        x = cp.Variable(n)
        constr = [b - A @ x >= 0]
        cp.Problem(cp.Minimize(x.T @ c), constr).solve()
        return x.value, constr[0].dual_value

    def _base_test_solvable(self, matrix, b, c):
        x, y = self.solve_program_cvxpy(matrix, b, c)
        self.check_solution_valid(matrix, b, c, x, y)
        # print('real solution x')
        # print(x)
        # print('real solution y')
        # print(y)
        solver = Solver(matrix, b, c, 0, len(b))
        self.check_solution_valid(matrix, b, c, solver.x, solver.y)

    def _base_test_infeasible_from_cvxpy(self, cvxpy_problem_obj):
        matrix, b, c = self.make_program_from_cvxpy(cvxpy_problem_obj)
        with self.assertRaises(Infeasible):
            solver = Solver(matrix, b, c, 0, len(b))
            cert = solver.y
            self.assertLess(b.T @ cert, 0)
            cert /= np.abs(b.T @ cert) # normalize
            self.assertGreater(np.min(cert), -1e-6)
            self.assertTrue(np.allclose(matrix.T @ cert, 0., atol=1e-6, rtol=1e-6))
            self.assertLess(b.T @ cert, 0)

    def _base_test_unbounded_from_cvxpy(self, cvxpy_problem_obj):
        matrix, b, c = self.make_program_from_cvxpy(cvxpy_problem_obj)
        with self.assertRaises(Unbounded):
            solver = Solver(matrix, b, c, 0, len(b))
            cert = solver.x
            print(cert)
            self.assertLess(c.T @ cert, 0)
            cert /= np.abs(c.T @ cert) # normalize
            conic = -matrix @ cert
            self.assertGreater(np.min(conic), -1e-6)
        # self.assertTrue(np.allclose(matrix.T @ cert, 0., atol=1e-6, rtol=1e-6))
        # self.assertLess(b.T @ cert, 0)

    def _base_test_solvable_from_cvxpy(self, cvxpy_problem_obj):
        matrix, b, c = self.make_program_from_cvxpy(cvxpy_problem_obj)
        self._base_test_solvable(matrix, b, c)

    def _base_test_solvable_from_matrix(self, matrix):
        b, c = self.make_program_from_matrix(matrix)
        self._base_test_solvable(matrix, b, c)

    def test_simple_infeasible(self):
        """Simple primal infeasible."""
        x = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(cp.norm1(x @ np.random.randn(5,10))),
            [x >= 0, x[3]<=-1.])
        self._base_test_infeasible_from_cvxpy(probl)


    def test_simple_unbounded(self):
        """Simple primal unbounded."""
        x = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(cp.norm1(x[1:] @ np.random.randn(4,10)) + x[0]),
            [x<=1.])
        self._base_test_unbounded_from_cvxpy(probl)

    def test_more_difficult_unbounded(self):
        """More difficult unbounded."""
        x = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(cp.sum(x @ np.random.randn(5,3))),
            [x<=1.])
        self._base_test_unbounded_from_cvxpy(probl)


    def test_more_difficult_infeasible(self):
        """More difficult primal infeasible."""
        np.random.randn(0)
        x = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(cp.norm1(x @ np.random.randn(5,10))),
            [np.random.randn(20,5) @ x >= 10])
        self._base_test_infeasible_from_cvxpy(probl)

    def test_from_cvxpy_redundant_constraints(self):
        """Test simple CVXPY problem with redundant constraints."""
        x = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(cp.norm1(x @ np.random.randn(5,10))),
            [x >= 0, x<=1.,  x<=1.]) # redundant constraints
        self._base_test_solvable_from_cvxpy(probl)

    def test_from_cvxpy_unused_variable(self):
        """Test simple CVXPY problem with unused variable."""
        x = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(cp.norm1(x[2:] @ np.random.randn(3,10))),
            [x[2:] >= 0, x[2:]<=1.])
        self._base_test_solvable_from_cvxpy(probl)

    def test_m_less_n_full_rank_(self):
        """m<n, matrix full rank."""
        np.random.seed(0)
        print('\nm<n, matrix full rank\n')
        matrix = np.random.randn(2, 5)
        self._base_test_solvable_from_matrix(matrix)


    def test_m_equal_n_full_rank_(self):
        """m=n, matrix full rank."""
        print('\nm=n, matrix full rank\n')
        np.random.seed(0)
        matrix = np.random.randn(3, 3)
        self._base_test_solvable_from_matrix(matrix)


    def test_m_greater_n_full_rank_(self):
        """m>n, matrix full rank."""
        np.random.seed(0)
        print('\nm>n, matrix full rank\n')
        matrix = np.random.randn(5, 2)
        self._base_test_solvable_from_matrix(matrix)


    def test_m_less_n_rank_deficient(self):
        """m<n, matrix rank deficient."""
        print('\nm<n, matrix rank deficient\n')
        np.random.seed(0)
        matrix = np.random.randn(2, 5)
        matrix = np.concatenate([matrix, [matrix.sum(0)]], axis=0)
        matrix = matrix[[0,2,1]]
        self._base_test_solvable_from_matrix(matrix)


    def test_m_equal_n_rank_deficient(self):
        """m=n, matrix rank deficient."""
        print('\nm=n, matrix rank deficient\n')
        np.random.seed(0)
        matrix = np.random.randn(2, 3)
        matrix = np.concatenate([matrix, [matrix.sum(0)]], axis=0)
        matrix = matrix[[0,2,1]]
        self._base_test_solvable_from_matrix(matrix)


    def test_m_greater_n_rank_deficient(self):
        """m>n, matrix rank deficient."""
        print('\nm>n, matrix rank deficient\n')
        np.random.seed(0)
        matrix = np.random.randn(5, 2)
        matrix = np.concatenate([matrix.T, [matrix.sum(1)]], axis=0).T
        # matrix = matrix[[0,2,1]]
        self._base_test_solvable_from_matrix(matrix)




    # def test(self):
    #     matrix = np.random.randn(2,5)
    #     breakpoint()
    #     b, c = self.make_program_from_matrix(matrix)
    #     x, y = self.solve_program_cvxpy(matrix, b, c)
    #     solver = Solver(matrix, b, c, 0, len(b))


if __name__ == '__main__': # pragma: no cover
    main()
