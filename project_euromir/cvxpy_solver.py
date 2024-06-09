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
    from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
        ConicSolver, NonNeg, Zero)
except ImportError as exc: # pragma: no cover
    raise ImportError(
        "Can't use CVXPY interface if CVXPY is not installed!") from exc


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

    def solve_via_data(self, *args, **kwargs):
        print("Solving with a custom QP solver!")
        super().solve_via_data(*args, **kwargs)
