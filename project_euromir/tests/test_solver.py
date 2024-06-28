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
            for seed in range(10):
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
                        ip_solver_stats[0][i]+np.finfo(float).eps)

                # self.assertTrue(
                #     np.allclose(my_solver_solution, ip_solver_solution))

                self.assertTrue(
                    np.isclose(my_solver_stats[1], ip_solver_stats[1]))


if __name__ == '__main__': # pragma: no cover
    import logging
    logging.basicConfig(level='INFO')
    main()
