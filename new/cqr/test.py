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
"""Unit tests of the solver class."""

import time
import warnings
from unittest import TestCase, main

import cvxpy as cp
import numpy as np
import scipy as sp

from .solver import Solver, Infeasible, Unbounded
from .cvxpy_interface import CQR


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

    def _y_in_cone(self, y, zero):
        self.assertTrue(
            # , atol=5e-5, rtol=1e-5)
            np.allclose(_dual_cone_project(y, zero), y)
        )

    def _s_in_cone(self, s, zero):
        self.assertTrue(
            np.allclose(_cone_project(s, zero), s)  # , atol=5e-5, rtol=1e-5)
        )

    def check_solution_valid(self, matrix, b, c, x, y, zero):
        """Check a cone program solution is valid."""
        # dual cone error
        print('DUAL CONE RESIDUAL NORM %.2e' % np.linalg.norm(
            _dual_cone_project(y, zero) - y))
        self._y_in_cone(y, zero)

        # primal cone error
        s = b - matrix @ x
        print('PRIMAL CONE RESIDUAL NORM %.2e' % np.linalg.norm(
            _cone_project(s, zero) - s))
        self._s_in_cone(s, zero)

        # gap error
        print('GAP RESIDUAL %.2e' % (c.T @ x + b.T @ y))
        self.assertTrue(
            np.isclose(c.T @ x, -b.T @ y)  # , atol=1e-6, rtol=1e-6)
        )
        # dual error
        print('DUAL RESIDUAL NORM %.2e' % np.linalg.norm(c + matrix.T @ y))
        self.assertTrue(
            np.allclose(c, - matrix.T @ y)  # , atol=1e-6, rtol=1e-6)
        )

    def check_infeasibility_certificate_valid(self, matrix, b, y, zero):
        """Check primal infeasibility certificate is valid."""
        y = np.copy(y)
        # TODO: this is here to pass tests, *remove* once refinement is there
        y[np.abs(y) < 1e-6] = 0.
        self.assertLess(b.T @ y, 0)
        y /= np.abs(b.T @ y)  # normalize
        self.assertTrue(np.isclose(b.T @ y, -1))
        # dual cone error
        self._y_in_cone(y, zero)
        # print(matrix.T @ y)
        self.assertTrue(
            np.allclose(matrix.T @ y, 0., atol=1e-6, rtol=1e-6)
        )

    def check_unboundedness_certificate_valid(self, matrix, c, x, zero):
        """Check primal unboundedness certificate is valid."""
        x = np.copy(x)
        self.assertLess(c.T @ x, 0)
        x /= np.abs(c.T @ x)  # normalize
        conic = -matrix @ x
        self._s_in_cone(conic, zero)

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
        # solver='SCS', verbose=True, acceleration_lookback=0)
        return program.status, x.value, constr[0].dual_value

    def check_solve(self, matrix, b, c, zero, nonneg, x0=None, y0=None):
        """Check solution or certificate is correct.

        We both check that CVXPY with default solver returns same status
        (optimal/infeasible/unbounded) and that the solution or
        certificate is valid. We don't look at the CVXPY solution or
        certificate (only the CVXPY status).
        """
        assert zero + nonneg == len(b)
        for qr in ['NUMPY', 'PYSPQR']:
            with self.subTest(qr=qr):
                solver = Solver(
                    sp.sparse.csc_matrix(matrix, copy=True),
                    np.array(b, copy=True), np.array(c, copy=True),
                    zero=zero, nonneg=nonneg, qr=qr, x0=x0, y0=y0)
                status, _, _ = self.solve_program_cvxpy(
                    sp.sparse.csc_matrix(matrix, copy=True),
                    np.array(b, copy=True), np.array(c, copy=True), zero=zero)
                if solver.status == 'Optimal':
                    self.assertIn(status, ['optimal', 'optimal_inaccurate'])
                    self.check_solution_valid(
                        matrix, b, c, solver.x, solver.y, zero=zero)
                elif solver.status == 'Infeasible':
                    self.assertIn(
                        status, ['infeasible', 'infeasible_inaccurate'])
                    self.check_infeasibility_certificate_valid(
                        matrix, b, solver.y, zero=zero)
                elif solver.status == 'Unbounded':
                    self.assertIn(
                        status, ['unbounded', 'unbounded_inaccurate'])
                    self.check_unboundedness_certificate_valid(
                        matrix, c, solver.x, zero=zero)
                else:
                    raise ValueError('Unknown solver status!')

        return solver.status, solver.x, solver.y,

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

    def _base_test_from_matrix(self, matrix, zero):
        assert zero <= matrix.shape[0]
        b, c = self.make_program_from_matrix(matrix, zero=zero)
        self.check_solve(matrix, b, c, zero=zero, nonneg=len(b)-zero)

    ###
    # Simple corner case tests
    ###

    def test_m_less_n_full_rank_(self):
        """M<n, matrix full rank."""
        np.random.seed(0)
        print('\nm<n, matrix full rank\n')
        matrix = np.random.randn(2, 5)
        for zero in range(matrix.shape[0]+1):
            self._base_test_from_matrix(matrix, zero=zero)

    def test_m_equal_n_full_rank_(self):
        """M=n, matrix full rank."""
        print('\nm=n, matrix full rank\n')
        np.random.seed(0)
        matrix = np.random.randn(3, 3)
        for zero in range(matrix.shape[0]+1):
            self._base_test_from_matrix(matrix, zero=zero)

    def test_m_greater_n_full_rank_(self):
        """M>n, matrix full rank."""
        np.random.seed(0)
        print('\nm>n, matrix full rank\n')
        matrix = np.random.randn(5, 2)
        for zero in range(matrix.shape[0]+1):
            self._base_test_from_matrix(matrix, zero=zero)

    def test_m_less_n_rank_deficient(self):
        """M<n, matrix rank deficient."""
        print('\nm<n, matrix rank deficient\n')
        np.random.seed(0)
        matrix = np.random.randn(2, 5)
        matrix = np.concatenate([matrix, [matrix.sum(0)]], axis=0)
        matrix = matrix[[0, 2, 1]]
        for zero in range(matrix.shape[0]+1):
            self._base_test_from_matrix(matrix, zero=zero)

    def test_m_equal_n_rank_deficient(self):
        """M=n, matrix rank deficient."""
        print('\nm=n, matrix rank deficient\n')
        np.random.seed(0)
        matrix = np.random.randn(2, 3)
        matrix = np.concatenate([matrix, [matrix.sum(0)]], axis=0)
        matrix = matrix[[0, 2, 1]]
        for zero in range(matrix.shape[0]+1):
            self._base_test_from_matrix(matrix, zero=zero)

    def test_m_greater_n_rank_deficient(self):
        """M>n, matrix rank deficient."""
        print('\nm>n, matrix rank deficient\n')
        np.random.seed(0)
        matrix = np.random.randn(5, 2)
        matrix = np.concatenate([matrix.T, [matrix.sum(1)]], axis=0).T
        # matrix = matrix[[0,2,1]]
        for zero in range(matrix.shape[0]+1):
            self._base_test_from_matrix(matrix, zero=zero)

    ###
    # Specify program as CVXPY object, reduce to code above
    ###

    @staticmethod
    def make_program_from_cvxpy(problem_obj):
        """Make program from cvxpy Problem object."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            data = problem_obj.get_problem_data('ECOS')[0]
        if data['A'] is None:
            matrix = data['G']
            b = data['h']
        if data['G'] is None:
            matrix = data['A']
            b = data['b']
        if (data['A'] is not None) and (data['G'] is not None):
            matrix = sp.sparse.vstack([data['A'], data['G']], format='csc')
            b = np.concatenate([data['b'], data['h']], dtype=float)
        return matrix, b, data['c'], data['dims'].zero, data['dims'].nonneg

    def check_solve_from_cvxpy(self, cvxpy_problem_obj):
        """Same as check solve, but takes CVXPY program object."""
        matrix, b, c, zero, nonneg = self.make_program_from_cvxpy(
            cvxpy_problem_obj)
        self.check_solve(matrix, b, c, zero, nonneg)

    ###
    # Check correct by specifying CVXPY programs
    ###

    def test_simple_redundant_x(self):
        """Check simple with redundant x variable."""
        x = cp.Variable(5)
        c = np.zeros(5)
        c[:-2] = 1.
        probl = cp.Problem(
            cp.Minimize(c.T @ x),
            [cp.abs(x) <= 1])
        self.check_solve_from_cvxpy(probl)
        probl = cp.Problem(
            cp.Minimize(c.T @ x),
            [x <= 1, x >= -1])
        self.check_solve_from_cvxpy(probl)

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
        np.random.seed(0)
        probl = cp.Problem(
            cp.Minimize(cp.sum(x @ np.random.randn(5, 3))),
            [x <= 1.])
        self.check_solve_from_cvxpy(probl)
        np.random.seed(1)
        probl = cp.Problem(
            cp.Minimize(cp.sum(x @ np.random.randn(5, 3))),
            [x <= 1., x[2] == 0.])
        self.check_solve_from_cvxpy(probl)
        np.random.seed(2)
        probl = cp.Problem(
            cp.Minimize(cp.sum(x @ np.random.randn(5, 3))),
            [x[:-1] <= 1., x[-1] == 0.])
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
            [x >= 0, x <= 1., x[2] == .5, x <= 1.])  # redundant constraints
        self.check_solve_from_cvxpy(probl)

    def test_from_cvxpy_unused_variable(self):
        """Test simple CVXPY problem with unused variable."""
        x = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(cp.norm1(x[2:] @ np.random.randn(3, 10))),
            [x[2:] >= 0, x[2:] <= 1.])
        self.check_solve_from_cvxpy(probl)

    def test_cvxpy_l1_regr(self):
        """Test simple l1 regression."""
        for i in range(10):
            np.random.seed(i)
            len_x = 5
            x = cp.Variable(len_x)

            A = np.random.randn(len_x*2, len_x)
            b = np.random.randn(len_x*2)
            probl = cp.Problem(
                cp.Minimize(cp.norm1(A @ x - b) + 0.01 * cp.norm1(x)),
                [cp.abs(x) <= 1.]
            )
            self.check_solve_from_cvxpy(probl)
            probl = cp.Problem(
                cp.Minimize(cp.norm1(A @ x - b) + 0.01 * cp.norm1(x)),
                [cp.abs(x) <= 1., x[1:-1] == 0.]
            )
            self.check_solve_from_cvxpy(probl)
            probl = cp.Problem(
                cp.Minimize(cp.norm1(A @ x - b) + 0.01 * cp.norm1(x)),
                [cp.abs(x) <= 1., x[1:-1] == 0., x[-1] >= 2]
            )
            self.check_solve_from_cvxpy(probl)

    def test_cvxpy_abs_transformation(self):
        """Test effect of expressing abs explicitely."""
        x = cp.Variable(5)
        c = np.random.randn(5)
        probl = cp.Problem(
            cp.Minimize(x.T @ c),
            [cp.abs(x) <= 1.]
        )
        self.check_solve_from_cvxpy(probl)
        probl = cp.Problem(
            cp.Minimize(x.T @ c),
            [x <= 1., x >= -1]
        )
        self.check_solve_from_cvxpy(probl)
        # first two give different programs in CVXPY
        probl = cp.Problem(
            cp.Minimize(x.T @ c),
            [cp.norm1(x) <= 1.]
        )
        self.check_solve_from_cvxpy(probl)
        y = cp.Variable(5)
        probl = cp.Problem(
            cp.Minimize(x.T @ c),
            [y >= x, y >= -x, cp.sum(y) <= 1]
        )
        self.check_solve_from_cvxpy(probl)
        # these instead give the same

    @staticmethod
    def _generate_problem_one(seed, m=41, n=30):
        """Generate a sample LP which can be difficult."""
        np.random.seed(seed)
        x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        objective = cp.norm1(A @ x - b)
        d = np.random.randn(n, 5)
        constraints = [cp.abs(x) <= .75, x @ d == 2.,]
        program = cp.Problem(cp.Minimize(objective), constraints)
        return x, program

    @staticmethod
    def _generate_problem_two(seed, m=70, n=40):
        """Generate a sample LP which can be difficult."""
        np.random.seed(seed)
        x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        objective = cp.norm1(A @ x - b) + 1. * cp.norm1(x)
        # adding these constraints, which are inactive at opt,
        # cause cg loop to stop early
        constraints = []  # x <= 1., x >= -1]
        program = cp.Problem(cp.Minimize(objective), constraints)
        return x, program

    def test_program_one(self):
        for seed in range(1):
            _, prog = self._generate_problem_one(seed)
            self.check_solve_from_cvxpy(prog)

    def test_program_two(self):
        for seed in range(1):
            _, prog = self._generate_problem_two(seed)
            self.check_solve_from_cvxpy(prog)

    def test_warmstart(self):
        """Simple test warmstart."""
        _, prog = self._generate_problem_one(seed=123, m=81, n=70)

        matrix, b, c, zero, nonneg = self.make_program_from_cvxpy(prog)

        s = time.time()
        status, x, y = self.check_solve(matrix, b, c, zero, nonneg)
        time_coldstart = time.time() - s

        x += np.random.randn(len(x)) * 1e-7
        y += np.random.randn(len(y)) * 1e-7

        s = time.time()
        self.check_solve(matrix, b, c, zero, nonneg, x0=x, y0=y)
        time_hotstart = time.time() - s

        self.assertLess(time_hotstart, time_coldstart)

    ###
    # Test CVXPY interface
    ###

    def test_cvxpy_interface(self):
        """Test correct translation to and from CVXPY."""
        x, prog = self._generate_problem_one(seed=321, m=20, n=10)
        prog.solve(solver=CQR())

        self.assertTrue(np.isposinf(cp.Problem(
            cp.Minimize(0.), [x >= 1, x <= 0]).solve(solver=CQR())))

        self.assertTrue(np.isneginf(cp.Problem(
            cp.Minimize(cp.sum(x)), [x <= 0]).solve(solver=CQR())))

if __name__ == '__main__':  # pragma: no cover
    main()
