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
"""Wrapper around real SCS."""

import scs
import numpy as np
import scipy as sp
import scs
from ..base_solver import BaseSolver
import tqdm

class RealSCS(BaseSolver):
    """Wrapper around real SCS.
    
    We use default parameters for SCS apart from convergence tolerances, which
    are made tighter. We also use our own convergence check to stop the loop
    of max_iters if we converge. Perhaps we could reliably push SCS to the
    convergence we use for the other prototypes - it's easy if we disable the
    acceleration code, making it much slower. Instead we use a laxer convergence
    tolerange (1e-8 instead of 1e-12). This seems a fair comparison, since I
    guess SCS is optimized to work well down to about that tolerance. Objective
    overall is to count number of (factorized) matrix multiplications, which
    is still 2 * num of iterations to convergence.
    """

    # class constants possibly overwritten by subclasses
    epsilon_convergence = 1e-8
    max_iterations = 100000 # outer loop of [100, 200, ..., 99_900, 100_000]
    scs_converged = 1e-12

    def loop(self):
        """Redefined full loop."""

        data = {
            "P": sp.sparse.csc_array((self.n, self.n)),
            "A": self.matrix, "b": self.b, "c": self.c}
        # assert len(self.soc) == 0
        cone = {"z": self.zero, "l": self.nonneg, "q": np.array(self.soc, dtype=int)}
        solver_iters = np.arange(
            0, int(self.max_iterations * (1.001)), self.max_iterations//1000)[1:]
        # for max_iters in tqdm.tqdm(solver_iters):
        for max_iters in solver_iters:
            # breakpoint()
            solver = scs.SCS(
                data, cone,
                verbose=False,
                max_iters=max_iters,
                # # uncomment the following to make it much slower but then it
                # # reaches 1e-12 on our scale in number of iters similar to
                # # non-accelerated CQR
                # acceleration_lookback=0,
                # adaptive_scale=False,
                # scale=1.,
                # rho_x=1.,
                # # alpha=1, # safe to leave to default
                eps_abs=self.scs_converged, eps_rel=self.scs_converged)
            result = solver.solve()
            # breakpoint()
            self.x = result['x']
            self.y = result['y']
            try:
                self.callback_iterate()
            except StopIteration:
                break
            if not "inaccurate" in result['info']['status']:
                # doing more would give us nothing
                break
        # print(self.solution_qualities)
        # import matplotlib.pyplot as plt
        # plt.semilogy(
        # solver_iters[:len(self.solution_qualities)], self.solution_qualities)
        # plt.show()
