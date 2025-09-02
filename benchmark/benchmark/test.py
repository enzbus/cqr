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

import gzip
import logging
import os

from unittest import TestCase, main, skip, skipIf

# pylint: disable=unused-import,unused-wildcard-import,wildcard-import

import cvxpy as cp
import numpy as np
import pandas as pd
import tqdm

from .cvxpy_interface import CvxpyWrapper
from .implementations.simple_bfgs import SimpleBFGS
from .implementations.simple_scs import *
from .implementations.simple_hsde import SimpleHSDE
from .implementations.simple_cqr import SimpleCQR
from .implementations.lm_scs import *
from .implementations.simple_cpr import SimpleCPR, EquilibratedCPR
from .implementations.new_cqr import *
from .implementations.real_scs import RealSCS

SOLVER_CLASS = os.getenv("SOLVER_CLASS")
NUM_INSTANCES = int(os.getenv("NUM_INSTANCES", "1"))
MODE = os.getenv("BENCHMARK_MODE", "BENCHMARK") # or "TEST"
SIZE_CHOICE = os.getenv("SIZE_CHOICE", "NORMAL") # or "SMALL"

PROGRAM_SIZES = {
    "NORMAL": {
        "_generate_problem_one": {"m": 41, "n": 30},
        "_generate_problem_two": {"m": 41, "n": 30},
        "_generate_portfolio_problem": {"n": 100}},
    "SMALL": {
        "_generate_problem_one": {"m": 4, "n": 3},
        "_generate_problem_two": {"m": 4, "n": 3},
        "_generate_portfolio_problem": {"n": 10}},
}
# logging.basicConfig(level='INFO')

class Benchmark(TestCase):
    """Unit tests of the solver class."""

    @staticmethod
    def _generate_problem_one(seed, m=41, n=30, ):
        """Generate a sample LP which can be difficult."""
        np.random.seed(seed)
        x = cp.Variable(n)
        mat = np.random.randn(m, n)
        b = np.random.randn(m)
        objective = cp.norm1(mat @ x - b)
        d = np.random.randn(n, (m+n)//14)
        constraints = [cp.abs(x) <= .75]
        if d.shape[1] > 0:
            constraints += [x @ d == 2.]
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
        eival = eival[-max(n//10, 1):]

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
        self._run_benchmark(self._generate_problem_one)

    # @skip("slow test, skip for now")
    def test_program_two(self):
        """Run second program class."""
        self._run_benchmark(self._generate_problem_two)

    # @skipIf(issubclass(globals()[SOLVER_CLASS], SimpleSHR),
    #     "SOCs not supported with those prototypes.")
    # @skip("slow test, skip for now")
    def test_po_program(self):
        """Run portf opt class."""
        self._run_benchmark(self._generate_portfolio_problem)

    def _run_benchmark(self, program_generator):
        """Run many instances, save history of solution qualities."""
        if MODE == "BENCHMARK":
            print('solver class', SOLVER_CLASS)
            solution_quality_curves = []
            print("PROGRAM", program_generator.__name__)
            for seed in tqdm.tqdm(range(NUM_INSTANCES)):
                _, prog = program_generator(
                    seed,
                    **PROGRAM_SIZES[SIZE_CHOICE][program_generator.__name__])
                prog.solve(solver=CvxpyWrapper(
                    solver_class=globals()[SOLVER_CLASS]))
                sol_qual = np.array(
                    prog.solver_stats.extra_stats['solution_qualities'])
                solution_quality_curves.append(sol_qual)

            # very rough
            sol_quals = pd.DataFrame(solution_quality_curves).T.ffill()

            # shouldn't get too heavy on disk
            with gzip.open(
                f"results/{SOLVER_CLASS}_"
                f"{program_generator.__name__.split('_generate_')[1]}.npy.gz",
                    "w") as f:
                np.save(file=f, arr=sol_quals.values)
        elif MODE == "TEST":
            all_prototypes = [
                el for el in globals().values()
                if (type(el) is type)
                and (not el is BaseSolver)
                and issubclass(el, BaseSolver)]
            for prototype in all_prototypes:
                print(prototype)
                prototype.max_iterations = 100
                with self.subTest(prototype=prototype):
                    _, prog = program_generator(0)
                    prog.solve(solver=CvxpyWrapper(
                        solver_class=prototype))
                    sol_qual = np.array(
                        prog.solver_stats.extra_stats['solution_qualities'])
                    # breakpoint()
                    self.assertLess(sol_qual[-1], sol_qual[0])
        else:
            # try integer single-seed mode
            print('solver class', SOLVER_CLASS)
            print("PROGRAM", program_generator.__name__)
            for seed in tqdm.tqdm([int(MODE)]):
                _, prog = program_generator(
                    seed,
                    **PROGRAM_SIZES[SIZE_CHOICE][program_generator.__name__])
                prog.solve(solver=CvxpyWrapper(
                    solver_class=globals()[SOLVER_CLASS]))
                sol_qual = np.array(
                    prog.solver_stats.extra_stats['solution_qualities'])
                breakpoint() # pylint: disable=forgotten-debug-statement

if __name__ == '__main__':  # pragma: no cover
    main()
