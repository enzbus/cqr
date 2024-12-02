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

import time

import numpy as np

import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    SOC, ConicSolver, NonNeg, Zero)
from cvxpy.error import SolverError

from .solver import Solver


class CQR(ConicSolver):
    """CVXPY interface for CQR.

    We follow SCS conventions: cones ordering, names in the dims dict, ...
    """

    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg, SOC]
    REQUIRES_CONSTR = False

    def import_solver(self):
        import cqr

    def name(self):
        return "CQR"

    def solve_via_data(
            self, data: dict, warm_start: bool, verbose: bool, solver_opts,
            solver_cache=None):
        """Main method."""

        solver = Solver(
            matrix=data['A'], b=data['b'], c=data['c'], zero=data['dims'].zero,
            nonneg=data['dims'].nonneg, soc=data['dims'].soc)

        return {
            'status': solver.status, 'value': np.dot(solver.x, data['c']),
            'x': solver.x, 'y': solver.y}

    def invert(self, solution, inverse_data):
        """CVXPY interface to propagate solution back."""

        attr = {}

        if solution['status'] == 'Optimal':

            status = s.OPTIMAL

            primal_val = solution["value"]
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

        elif solution['status'] == 'Infeasible':
            attr[s.EXTRA_STATS] = {
                'infeasibility_certificate': solution['y']
            }
            status = s.INFEASIBLE
            return failure_solution(status, attr)

        elif solution['status'] == 'Unbounded':
            attr[s.EXTRA_STATS] = {
                'unboundedness_certificate': solution['x']
            }
            status = s.UNBOUNDED
            return failure_solution(status, attr)

        else:
            raise SolverError('Unknown solver status!')
