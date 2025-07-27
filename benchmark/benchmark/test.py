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

import logging
from unittest import TestCase, main

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from .cvxpy_interface import CvxpyWrapper
from .implementations.simple_bfgs import SimpleBFGS
from .implementations.simple_scs import SimpleSCS


logging.basicConfig(level='INFO')

class Benchmark(TestCase):
    """Unit tests of the solver class."""

    @staticmethod
    def _generate_problem_one(seed, m=41, n=30):
        """Generate a sample LP which can be difficult."""
        np.random.seed(seed)
        x = cp.Variable(n)
        mat = np.random.randn(m, n)
        b = np.random.randn(m)
        objective = cp.norm1(mat @ x - b)
        d = np.random.randn(n, 5)
        constraints = [cp.abs(x) <= .75, x @ d == 2.,]
        program = cp.Problem(cp.Minimize(objective), constraints)
        return x, program

    @staticmethod
    def _generate_problem_two(seed, m=41, n=30):
        """Generate a sample LP which can be difficult."""
        np.random.seed(seed)
        x = cp.Variable(n)
        mat = np.random.randn(m, n)
        b = np.random.randn(m)
        objective = cp.norm1(mat @ x - b) + 1. * cp.norm1(x)
        # adding these constraints, which are inactive at opt,
        # cause cg loop to stop early
        constraints = []  # x <= 1., x >= -1]
        program = cp.Problem(cp.Minimize(objective), constraints)
        return x, program

    @staticmethod
    def _generate_portfolio_problem(seed, n=100):
        np.random.seed(seed)
        w = cp.Variable(n)
        w0 = np.random.randn(n)
        w0 -= np.sum(w0)/len(w0)
        w0 /= np.sum(np.abs(w0))
        mu = np.random.randn(n) * 1e-3
        big_sigma = np.random.randn(n, n)
        big_sigma = big_sigma.T @ big_sigma
        eival, eivec = np.linalg.eigh(big_sigma)
        eival *= 1e-4
        eival = eival[-n//10:]

        # make it feasible; reduce w0 size so that it's in risk cone
        risk = cp.sum_squares((np.diag(np.sqrt(eival))
                @ eivec[:, -n//10:].T) @ w)
        risk_limit = 0.00005

        for _ in range(10):
            w.value = w0
            if risk.value < risk_limit:
                break
            w0 /= 2.
        else:
            raise ValueError("Increase counter, wasn't enough.")

        # Sigma = eivec @ np.diag(eival) @ eivec.T
        objective = w.T @ mu + 1e-5 * cp.norm1(w-w0)
        constraints = [#w >=0, #w<=w_max,
            cp.sum(w) == 0, cp.norm1(w-w0) <= 0.05,
            cp.norm1(w) <= 1,
            risk <= risk_limit]
        program = cp.Problem(cp.Minimize(objective), constraints)
        # program.solve(solver='SCS', verbose=True, eps=1e-14)
        return w, program

    # @skip("slow test, skip for now")
    def test_program_one(self):
        """Run first program class."""
        for seed in range(1):
            _, prog = self._generate_problem_one(seed)
            self.solve_program(prog)

    # @skip("slow test, skip for now")
    def test_program_two(self):
        """Run second program class."""
        solution_quality_curves = []
        for seed in range(200):
            _, prog = self._generate_problem_two(seed)
            solution_quality_curves.append(self.solve_program(prog))
            plt.semilogy(solution_quality_curves[-1])
        plt.show()
        # import matplotlib.pyplot as plt
        # plt.semilogy(sol_qual)
        # plt.show()

    # @skip("slow test, skip for now")
    def test_po_program(self):
        """Run portf opt class."""
        solution_quality_curves = []
        for seed in range(200):
            _, prog = self._generate_portfolio_problem(seed)
            solution_quality_curves.append(self.solve_program(prog))
            plt.semilogy(solution_quality_curves[-1])
        plt.show()
        # import matplotlib.pyplot as plt
        # plt.semilogy(sol_qual)
        # plt.show()

    def solve_program(self, prog):
        """Solve given CVXPY program.
        
        :param prog: CVXPY Problem object.
        :type prog: cp.Problem
        """
        for solver_class in [
                #SimpleBFGS
                SimpleSCS]:
            print('solver class', solver_class)
            prog.solve(solver=CvxpyWrapper(solver_class=solver_class))
            sol_qual = np.array(
                prog.solver_stats.extra_stats['solution_qualities'])
            return sol_qual
            # import matplotlib.pyplot as plt
            # plt.semilogy(sol_qual)
            # plt.show()

if __name__ == '__main__':  # pragma: no cover
    main()
