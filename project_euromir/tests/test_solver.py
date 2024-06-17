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
from unittest import TestCase, main

import cvxpy as cp
import numpy as np

from project_euromir.cvxpy_solver import Solver


class TestSolver(TestCase):
    """Test solver."""

    def test_simple(self):
        """Test on simple LP."""
        np.random.seed(0)
        m, n = 21, 20
        x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        objective = cp.norm1(A @ x - b)
        d = np.random.randn(n)
        constraints = [cp.abs(x) <= .5, x @ d == 1.,]
        s = time.time()
        cp.Problem(cp.Minimize(objective), constraints).solve(solver=Solver())
        print('PROTOTYPE SOLVER TOOK', time.time() - s)
        self.assertTrue(np.isclose(np.max(np.abs(x.value)), .5))
        project_euromir_solution = x.value
        print(f'CONSTRAINTS VIOLATION NORMs, solver: {np.linalg.norm(constraints[0].violation()):e}, {np.linalg.norm(constraints[1].violation()):e}')

        s = time.time()
        cp.Problem(cp.Minimize(objective), constraints).solve(
            solver='ECOS', feastol = 1e-16, abstol = 1e-16, reltol = 1e-16, verbose=True)
        print('INTERIOR POINT TOOK', time.time() - s)
        print(f'CONSTRAINTS VIOLATION NORM, INTERIOR POINT: {np.linalg.norm(constraints[0].violation()):e}, {np.linalg.norm(constraints[1].violation()):e}')

        ip_solution = x.value

        pe = np.sum(np.abs(A @ project_euromir_solution - b))
        ip = np.sum(np.abs(A @ ip_solution - b))

        print(f'Objective value, solver: {pe:e}')
        print(f'Objectve value, INTERIOR POINT: {ip:e}')
        print(f'ProjEur - IP objective vals {pe-ip:e}')

        # breakpoint()

        # print(project_euromir_solution)
        # print(clarabel_solution)

        self.assertTrue(
            np.allclose(project_euromir_solution, ip_solution))

        self.assertTrue(np.isclose(pe, ip))


if __name__ == '__main__': # pragma: no cover
    import logging
    logging.basicConfig(level='INFO')
    main()
