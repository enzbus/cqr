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

    def test_simple(self):
        """Test on simple LP."""
        for seed in range(10):
            print('\n\nEXPERIMENT', seed+1)
            np.random.seed(seed)
            m, n = 21, 20
            x = cp.Variable(n)
            A = np.random.randn(m, n)
            b = np.random.randn(m)
            objective = cp.norm1(A @ x - b)
            d = np.random.randn(n, 2)
            constraints = [cp.abs(x) <= .5, x @ d == 1.,]

            def _get_stats():
                constr_err_1 = np.linalg.norm(constraints[0].violation())
                constr_err_2 = np.linalg.norm(constraints[1].violation())
                return constr_err_1, constr_err_2, objective.value

            s = time.time()
            cp.Problem(
                cp.Minimize(objective), constraints).solve(solver=Solver())
            print('PROTOTYPE SOLVER TOOK', time.time() - s)
            my_solver_solution = x.value
            my_solver_stats = _get_stats()
            print(
                'PROTOTYPE SOLVER STATS; constr violation norms: '
                f'({my_solver_stats[0]:.2e}, {my_solver_stats[1]:.2e}). '
                f'objective: {my_solver_stats[2]:.2e}')

            s = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cp.Problem(cp.Minimize(objective), constraints).solve(
                    solver='ECOS', feastol = 1e-32, abstol = 1e-32,
                    reltol = 1e-32, max_iters=1000)#, verbose=True)
            print('INTERIOR POINT TOOK', time.time() - s)
            ip_solver_solution = x.value
            ip_solver_stats = _get_stats()
            print(
                'INTERIOR POINT SOLVER STATS; constr violation norms: '
                f'({ip_solver_stats[0]:.2e}, {ip_solver_stats[1]:.2e}), '
                f'objective: {ip_solver_stats[2]:.2e}')

            print('Prototype objective - IP objective: '
                f'{my_solver_stats[2]-ip_solver_stats[2]:e}')

            # we just add float epsilon for the LP constraint
            self.assertLessEqual(
                my_solver_stats[0], ip_solver_stats[0]+np.finfo(float).eps)

            # this is the equality constraint, seems more sensitive to scaling
            # we add float epsilon with multiplier b/c fails on other platforms
            self.assertLessEqual(
                my_solver_stats[1], ip_solver_stats[1]+5*np.finfo(float).eps)

            self.assertTrue(
                np.allclose(my_solver_solution, ip_solver_solution))

            self.assertTrue(np.isclose(my_solver_stats[2], ip_solver_stats[2]))


if __name__ == '__main__': # pragma: no cover
    import logging
    logging.basicConfig(level='INFO')
    main()
