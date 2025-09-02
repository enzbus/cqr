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
    """Wrapper around real SCS."""

    # class constants possibly overwritten by subclasses
    epsilon_convergence = 1e-12
    max_iterations = 100000

    def loop(self):
        """Redefined full loop."""

        data = {
            "P": sp.sparse.csc_array((self.n, self.n)),
            "A": self.matrix, "b": self.b, "c": self.c}
        # assert len(self.soc) == 0
        cone = {"z": self.zero, "l": self.nonneg, "q": np.array(self.soc)}
        solver_iters = np.arange(0, 101000, 1000)[1:]
        # for max_iters in tqdm.tqdm(solver_iters):
        for max_iters in solver_iters:
            solver = scs.SCS(
                data, cone,
                verbose=False,
                max_iters=max_iters,
                acceleration_lookback=0,
                adaptive_scale=False,
                scale=1.,
                rho_x=1.,
                # alpha=1, # safe to leave to default
                eps_abs=1e-16, eps_rel=1e-16)
            result = solver.solve()
            self.x = result['x']
            self.y = result['y']
            try:
                self.callback_iterate()
            except StopIteration:
                break
        # print(self.solution_qualities)
        # import matplotlib.pyplot as plt
        # plt.semilogy(
        # solver_iters[:len(self.solution_qualities)], self.solution_qualities)
        # plt.show()
