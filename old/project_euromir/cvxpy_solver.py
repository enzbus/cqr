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
"""CVXPY solver interface.

Documentation:
    https://www.cvxpy.org/tutorial/solvers/index.html#custom-solvers

Relevant base class:
    https://github.com/cvxpy/cvxpy/blob/master/cvxpy/reductions/solvers/conic_solvers/conic_solver.py

Model class:
    https://github.com/cvxpy/cvxpy/blob/master/cvxpy/reductions/solvers/conic_solvers/scs_conif.py
"""

import time

import numpy as np

try:
    import cvxpy.settings as s
    from cvxpy.reductions.solution import Solution
    from cvxpy.reductions.solvers import utilities
    from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
        SOC, ConicSolver, NonNeg, Zero)
except ImportError as exc: # pragma: no cover
    raise ImportError(
        "Can't use CVXPY interface if CVXPY is not installed!") from exc

# original, with hsde
# from project_euromir import solve

BFGS = False
NEWTON_CG = False
NEW = True
assert (BFGS + NEWTON_CG + NEW) == 1

if BFGS:
    # without hsde
    from project_euromir.solver_nohsde import solve

if NEWTON_CG:

    # newton cg
    from project_euromir.solver_cg import solve

if NEW:

    # reimplementation of newton_cg outside of scipy
    from project_euromir.solver_new import solve

# from project_euromir.solver_dense_solve import solve


class Solver(ConicSolver):
    """CVXPY solver interface.

    We follow SCS conventions for simplicity: cones ordering, in the future
    exp and sdp cone definition, names in the dims dict (SCS 3), ...
    """

    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg, SOC]
    REQUIRES_CONSTR = False

    def import_solver(self):
        import project_euromir

    def name(self):
        return "PROJECT_EUROMIR"

    def solve_via_data(
            self, data: dict, warm_start: bool, verbose: bool, solver_opts,
            solver_cache=None):
        """Main method."""

        x_orig, y_orig, s_orig = solve(
            matrix=data['A'], b=data['b'], c=data['c'], zero=data['dims'].zero,
            nonneg=data['dims'].nonneg, soc=data['dims'].soc)

        # breakpoint()

        return {
            'primal_val': np.dot(x_orig, data['c']), 'x': x_orig, 'y': y_orig,
            's': s_orig, 'status': 'OPTIMAL'}

    def invert(self, solution, inverse_data):
        """CVXPY interface to propagate solution back."""

        status = s.OPTIMAL

        attr = {}
        primal_val = solution["primal_val"]
        opt_val = primal_val + inverse_data[s.OFFSET]
        primal_vars = {
            inverse_data[self.VAR_ID]: solution["x"]
        }
        eq_dual_vars = utilities.get_dual_values(
            solution["y"][:inverse_data[ConicSolver.DIMS].zero],
            utilities.extract_dual_value,
            inverse_data[self.EQ_CONSTR]
        )
        ineq_dual_vars = utilities.get_dual_values(
            solution["y"][inverse_data[ConicSolver.DIMS].zero:],
            utilities.extract_dual_value,
            inverse_data[self.NEQ_CONSTR]
        )
        dual_vars = {}
        dual_vars.update(eq_dual_vars)
        dual_vars.update(ineq_dual_vars)
        return Solution(status, opt_val, primal_vars, dual_vars, attr)
