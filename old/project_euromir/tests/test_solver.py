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
"""Simple testing of the CVXPY solver interface."""

import time
import warnings
from unittest import TestCase, main

import cvxpy as cp
import numpy as np

from project_euromir.cvxpy_solver import Solver


class TestSolver(TestCase):
    """Test solver."""

    @staticmethod
    def _generate_problem_one(seed):
        np.random.seed(seed)
        m, n = 81, 70
        x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        objective = cp.norm1(A @ x - b)
        d = np.random.randn(n, 5)
        constraints = [cp.abs(x) <= .75, x @ d == 2.,]
        program = cp.Problem(cp.Minimize(objective), constraints)
        return x, program

    @staticmethod
    def _generate_problem_two(seed):
        np.random.seed(seed)
        m, n = 150, 70
        x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        objective = cp.norm1(A @ x - b) + 1. * cp.norm1(x)
        # adding these constraints, which are inactive at opt,
        # cause cg loop to stop early
        constraints = []#x <= 1., x >= -1]
        program = cp.Problem(cp.Minimize(objective), constraints)
        return x, program

    def test_simple(self):
        """Test on simple LP."""
        for generator in [
                self._generate_problem_one,
                self._generate_problem_two]:
            for seed in range(10): #(27,29):#100): #(405, 407):
                print('\n\nEXPERIMENT', seed+1)
                x, program = generator(
                    seed)

                def _get_stats(program):
                    constr_errs = [
                        np.linalg.norm(constraint.violation())
                            for constraint in program.constraints]
                    return constr_errs, program.objective.value

                s = time.time()
                program.solve(solver=Solver())
                print('PROTOTYPE SOLVER TOOK', time.time() - s)
                my_solver_solution = x.value
                my_solver_stats = _get_stats(program)
                print(
                    'PROTOTYPE SOLVER STATS; constr violation norms: ',
                    my_solver_stats[0],
                    #f'({my_solver_stats[0]:.2e}, {my_solver_stats[1]:.2e}). '
                    f'objective: {my_solver_stats[1]:.2e}')

                s = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    program.solve(
                        solver='ECOS', feastol = 1e-32, abstol = 1e-32,
                        reltol = 1e-32, max_iters=1000)#, verbose=True)
                print('INTERIOR POINT TOOK', time.time() - s)
                ip_solver_solution = x.value
                ip_solver_stats = _get_stats(program)
                print(
                    'INTERIOR POINT SOLVER STATS; constr violation norms: ',
                    ip_solver_stats[0],
                    # f'({ip_solver_stats[0]:.2e}, {ip_solver_stats[1]:.2e}), '
                    f'objective: {ip_solver_stats[1]:.2e}')

                print('Prototype objective - IP objective: '
                    f'{my_solver_stats[1]-ip_solver_stats[1]:e}')

                # we just add float epsilon for the LP constraints
                for i in range(len(program.constraints)):
                    self.assertLessEqual(
                        my_solver_stats[0][i],
                        # might have to increase this
                        ip_solver_stats[0][i]+5 * np.finfo(float).eps)

                # self.assertTrue(
                #     np.allclose(my_solver_solution, ip_solver_solution))

                self.assertTrue(
                    np.isclose(my_solver_stats[1], ip_solver_stats[1]))


if __name__ == '__main__': # pragma: no cover
    import logging
    logging.basicConfig(level='INFO')
    main()
