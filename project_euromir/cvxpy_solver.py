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

try:
    import cvxpy.settings as s
    from cvxpy.reductions.solution import Solution
    from cvxpy.reductions.solvers import utilities
    from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
        ConicSolver, NonNeg, Zero)
except ImportError as exc: # pragma: no cover
    raise ImportError(
        "Can't use CVXPY interface if CVXPY is not installed!") from exc

import numpy as np
import scipy as sp

from project_euromir import equilibrate


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

        matrix = data['A']
        b = data['b']
        c = data['c']
        n = len(c)
        m = len(b)
        assert matrix.shape == (m, n)
        zero = data['dims'].zero
        nonneg = data['dims'].nonneg
        assert zero + nonneg == m

        d, e, sigma, rho, matrix_transf, b_transf, c_transf = \
            equilibrate.hsde_ruiz_equilibration(
                    matrix, b, c, dimensions={
                        'zero': zero, 'nonneg': nonneg, 'second_order': ()},
                    max_iters=100)

        Q = sp.sparse.bmat([
            [None, matrix_transf.T, c_transf.reshape(n, 1)],
            [-matrix_transf, None, b_transf.reshape(m, 1)],
            [-c_transf.reshape(1, n), -b_transf.reshape(1, m), None],
            ]).tocsc()

        # [Q, -I]
        QI = sp.sparse.hstack([Q, -sp.sparse.eye(n+m+1, format='csc')])

        # remove v in zero cone
        _as = np.concatenate(
            [np.ones(n+m+1, dtype=bool),
            np.zeros(n + zero, dtype=bool),
            np.ones(m+1 - zero, dtype=bool)])
        system_matrix = QI[:, _as]

        def loss_gradient(variable):
            residual = system_matrix @ variable
            err = np.minimum(variable[n:], 0)
            loss = np.linalg.norm(residual)**2 + np.linalg.norm(err)**2
            grad = 2 * (system_matrix.T @ residual)
            grad[n:] += 2 * err
            return loss, grad

        x_0 = np.zeros(system_matrix.shape[1])
        x_0[n+m] = 1.

        lbfgs_result = sp.optimize.fmin_l_bfgs_b(
            loss_gradient,
            x0=x_0,
            m=10,
            maxfun=1e10,
            factr=0., pgtol=0.,
            maxiter=1e10)

        u = lbfgs_result[0][:n+m+1]
        v = np.zeros(n+m+1)
        v[n+zero:] = lbfgs_result[0][n+m+1:]

        # TODO: LSQR goes here

        u1, u2, u3 = u[:n], u[n:n+m], u[-1]
        v2, v3 = v[n:n+m], v[-1]

        if v3 > u3:
            raise NotImplementedError('Certificates not yet implemented.')

        # Apply HSDE scaling
        x = u1 / u3
        y = u2 / u3
        s = v2 / u3

        # invert Ruiz scaling, copied from other repo
        x_orig =  e * x / sigma
        y_orig = d * y / rho
        s_orig = (s/d) / sigma

        return {
            'primal_val': np.dot(x_orig, c), 'x': x_orig, 'y': y_orig,
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
