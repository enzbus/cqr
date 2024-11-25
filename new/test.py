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


# Simple implementation of cone projection for tests

def _cone_project(s, zero):
    """Project on program cone."""
    return np.concatenate([
        np.zeros(zero), np.maximum(s[zero:], 0.)])

def _dual_cone_project(y, zero):
    """Project on dual of program cone."""
    return np.concatenate([
        y[:zero], np.maximum(y[zero:], 0.)])

class TestSolverClass(TestCase):
    """Unit tests of the solver class."""

    ###
    # Logic to check solution or certificate validity
    ###

    def check_solution_valid(self, matrix, b, c, x, y, zero):
        """Check a cone program solution is valid."""
        # dual cone error
        self.assertTrue(
            np.allclose(_dual_cone_project(y, zero), y)
        )
        s = b - matrix @ x
        # primal cone error
        self.assertTrue(
            np.allclose(_cone_project(s, zero), s)
        )
        # gap error
        self.assertTrue(
            np.isclose(c.T @ x, -b.T @ y) #, atol=1e-6, rtol=1e-6)
            )
        # dual error
        self.assertTrue(
            np.allclose(c, - matrix.T @ y) #, atol=1e-6, rtol=1e-6)
            )

    def check_infeasibility_certificate_valid(self, matrix, b, y, zero):
        """Check primal infeasibility certificate is valid."""
        y = np.copy(y)
        self.assertLess(b.T @ y, 0)
        y /= np.abs(b.T @ y) # normalize
        self.assertTrue(np.isclose(b.T @ y, -1))
        # dual cone error
        self.assertTrue(
            np.allclose(_dual_cone_project(y, zero), y)
        )
        self.assertTrue(
            np.allclose(matrix.T @ y, 0.)#, atol=1e-6, rtol=1e-6)
            )

    def check_unboundedness_certificate_valid(self, matrix, c, x, zero):
        """Check primal unboundedness certificate is valid."""
        x = np.copy(x)
        self.assertLess(c.T @ x, 0)
        x /=  np.abs(c.T @ x) # normalize
        conic = -matrix @ x
        self.assertTrue(
            np.allclose(_cone_project(conic, zero), conic)
        )

    @staticmethod
    def solve_program_cvxpy(matrix, b, c, zero):
        """Solve simple LP with CVXPY."""
        m, n = matrix.shape
        x = cp.Variable(n)
        constr = []
        if zero > 0:
            constr.append(b[:zero] - matrix[:zero] @ x == 0)
        if zero < len(b):
            constr.append(b[zero:] - matrix[zero:] @ x >= 0)
        program = cp.Problem(cp.Minimize(x.T @ c), constr)
        program.solve()
        return program.status, x.value, constr[0].dual_value

    def check_solve(self, matrix, b, c, zero, nonneg):
        """Check solution or certificate is correct.

        We both check that CVXPY with default solver returns same status
        (optimal/infeasible/unbounded) and that the solution or
        certificate is valid. We don't look at the CVXPY solution or
        certificate.
        """
        assert zero + nonneg == len(b)
        solver = Solver(
            sp.sparse.csc_matrix(matrix, copy=True),
            np.array(b, copy=True), np.array(c, copy=True),
            zero=zero, nonneg=nonneg)
        status, _, _ = self.solve_program_cvxpy(
            sp.sparse.csc_matrix(matrix, copy=True),
            np.array(b, copy=True), np.array(c, copy=True), zero=zero)
        if solver.status == 'Optimal':
            self.assertEqual(status, 'optimal')
            self.check_solution_valid(matrix, b, c, solver.x, solver.y, zero=zero)
        elif solver.status == 'Infeasible':
            self.assertEqual(status, 'infeasible')
            self.check_infeasibility_certificate_valid(matrix, b, solver.y, zero = zero)
        elif solver.status == 'Unbounded':
            self.assertEqual(status, 'unbounded')
            self.check_unboundedness_certificate_valid(matrix, c, solver.x, zero=zero)
        else:
            raise ValueError('Unknown solver status!')

    ###
    # Logic to create program and check it
    ###

    @staticmethod
    def make_program_from_matrix(matrix, zero, seed=0):
        """Make simple LP program."""
        m, n = matrix.shape
        np.random.seed(seed)
        z = np.random.randn(m)
        y = _dual_cone_project(z, zero)
        s = y - z
        x = np.random.randn(n)
        b = matrix @ x + s
        c = -matrix.T @ y
        return b, c

    def _base_test_solvable_from_matrix(self, matrix):
        b, c = self.make_program_from_matrix(matrix, zero=0)
        self.check_solve(matrix, b, c, zero=0, nonneg=len(b))

    ###
    # Simple corner case tests
    ###

    def test_m_less_n_full_rank_(self):
        """M<n, matrix full rank."""
        np.random.seed(0)
        print('\nm<n, matrix full rank\n')
        matrix = np.random.randn(2, 5)
        self._base_test_solvable_from_matrix(matrix)

    def test_m_equal_n_full_rank_(self):
        """M=n, matrix full rank."""
        print('\nm=n, matrix full rank\n')
        np.random.seed(0)
        matrix = np.random.randn(3, 3)
        self._base_test_solvable_from_matrix(matrix)

    def test_m_greater_n_full_rank_(self):
        """M>n, matrix full rank."""
        np.random.seed(0)
        print('\nm>n, matrix full rank\n')
        matrix = np.random.randn(5, 2)
        self._base_test_solvable_from_matrix(matrix)

    def test_m_less_n_rank_deficient(self):
        """M<n, matrix rank deficient."""
        print('\nm<n, matrix rank deficient\n')
        np.random.seed(0)
        matrix = np.random.randn(2, 5)
        matrix = np.concatenate([matrix, [matrix.sum(0)]], axis=0)
        matrix = matrix[[0, 2, 1]]
        self._base_test_solvable_from_matrix(matrix)

    def test_m_equal_n_rank_deficient(self):
        """M=n, matrix rank deficient."""
        print('\nm=n, matrix rank deficient\n')
        np.random.seed(0)
        matrix = np.random.randn(2, 3)
        matrix = np.concatenate([matrix, [matrix.sum(0)]], axis=0)
        matrix = matrix[[0, 2, 1]]
        self._base_test_solvable_from_matrix(matrix)

    def test_m_greater_n_rank_deficient(self):
        """M>n, matrix rank deficient."""
        print('\nm>n, matrix rank deficient\n')
        np.random.seed(0)
        matrix = np.random.randn(5, 2)
        matrix = np.concatenate([matrix.T, [matrix.sum(1)]], axis=0).T
        # matrix = matrix[[0,2,1]]
        self._base_test_solvable_from_matrix(matrix)

    ###
    # Specify program as CVXPY object, reduce to code above
    ###

    @staticmethod
    def make_program_from_cvxpy(problem_obj):
        """Make program from cvxpy Problem object."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            data = problem_obj.get_problem_data('ECOS')[0]
        assert data['dims'].zero == 0 # will need to update base methods
        # the eq components are data['A'] and data['b']
        return data['G'], data['h'], data['c'], data['dims'].zero, data['dims'].nonneg

    def check_solve_from_cvxpy(self, cvxpy_problem_obj):
        """Same as check solve, but takes CVXPY program object."""
        matrix, b, c, zero, nonneg = self.make_program_from_cvxpy(cvxpy_problem_obj)
        self.check_solve(matrix, b, c, zero, nonneg)

    ###
    # Check correct by specifying CVXPY programs
    ###

    def test_simple_infeasible(self):
        """Simple primal infeasible."""
        x = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(cp.norm1(x @ np.random.randn(5, 10))),
            [x >= 0, x[3] <= -1.])
        self.check_solve_from_cvxpy(probl)

    def test_simple_unbounded(self):
        """Simple primal unbounded."""
        x = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(cp.norm1(x[1:] @ np.random.randn(4, 10)) + x[0]),
            [x <= 1.])
        self.check_solve_from_cvxpy(probl)

    def test_more_difficult_unbounded(self):
        """More difficult unbounded."""
        x = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(cp.sum(x @ np.random.randn(5, 3))),
            [x <= 1.])
        self.check_solve_from_cvxpy(probl)

    def test_more_difficult_infeasible(self):
        """More difficult primal infeasible."""
        np.random.randn(0)
        x = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(cp.norm1(x @ np.random.randn(5, 10))),
            [np.random.randn(20, 5) @ x >= 10])
        self.check_solve_from_cvxpy(probl)

    def test_from_cvxpy_redundant_constraints(self):
        """Test simple CVXPY problem with redundant constraints."""
        x = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(cp.norm1(x @ np.random.randn(5, 10))),
            [x >= 0, x <= 1.,  x <= 1.]) # redundant constraints
        self.check_solve_from_cvxpy(probl)

    def test_from_cvxpy_unused_variable(self):
        """Test simple CVXPY problem with unused variable."""
        x = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(cp.norm1(x[2:] @ np.random.randn(3, 10))),
            [x[2:] >= 0, x[2:] <= 1.])
        self.check_solve_from_cvxpy(probl)

    # def test(self):
    #     matrix = np.random.randn(2,5)
    #     breakpoint()
    #     b, c = self.make_program_from_matrix(matrix)
    #     x, y = self.solve_program_cvxpy(matrix, b, c)
    #     solver = Solver(matrix, b, c, 0, len(b))


if __name__ == '__main__': # pragma: no cover
    main()
