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
        ConicSolver, NonNeg, Zero)
except ImportError as exc: # pragma: no cover
    raise ImportError(
        "Can't use CVXPY interface if CVXPY is not installed!") from exc

# original, with hsde
# from project_euromir import solve

BFGS = False
NEWTON_CG = not BFGS

if BFGS:
    # without hsde
    from project_euromir.solver_nohsde import solve

if NEWTON_CG:

    # newton cg
    from project_euromir.solver_cg import solve

# from project_euromir.solver_dense_solve import solve


class Solver(ConicSolver):
    """CVXPY solver interface.

    We follow SCS conventions for simplicity: cones ordering, in the future
    exp and sdp cone definition, names in the dims dict (SCS 3), ...
    """

    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg]
    REQUIRES_CONSTR = False

    def name(self):
        return "PROJECT_EUROMIR"

    def solve_via_data(
            self, data: dict, warm_start: bool, verbose: bool, solver_opts,
            solver_cache=None):
        """Main method."""

        x_orig, y_orig, s_orig = solve(
            matrix=data['A'], b=data['b'], c=data['c'], zero=data['dims'].zero,
            nonneg=data['dims'].nonneg)

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
