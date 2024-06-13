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
        m, n = 20, 20
        x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        objective = cp.norm1(A @ x - b)
        constraints = [cp.abs(x) <= .5]
        s = time.time()
        cp.Problem(cp.Minimize(objective), constraints).solve(solver=Solver())
        print('PROTOTYPE SOLVER TOOK', time.time() - s)
        self.assertTrue(np.isclose(np.max(np.abs(x.value)), .5))
        project_euromir_solution = x.value

        s = time.time()
        cp.Problem(cp.Minimize(objective), constraints).solve(
            solver='CLARABEL', verbose=True)
        print('CLARABEL TOOK', time.time() - s)

        clarabel_solution = x.value

        pe = np.sum(np.abs(A @ project_euromir_solution - b))
        clarabel = np.sum(np.abs(A @ clarabel_solution - b))

        print(pe)
        print(clarabel)

        print(project_euromir_solution)
        print(clarabel_solution)

        self.assertTrue(
            np.allclose(project_euromir_solution, clarabel_solution))

        self.assertTrue(np.isclose(pe, clarabel))


if __name__ == '__main__': # pragma: no cover
    import logging
    logging.basicConfig(level='INFO')
    main()
