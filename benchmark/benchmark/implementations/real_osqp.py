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
"""Wrapper around real OSQP."""

import osqp
import numpy as np
import scipy as sp
from ..base_solver import BaseSolver
import tqdm

class RealOSQP(BaseSolver):
    """Wrapper around real OSQP."""

    # class constants possibly overwritten by subclasses
    epsilon_convergence = 1e-12
    max_iterations = 100000

    def loop(self):
        """Redefined full loop."""

        assert len(self.soc) == 0, "SOCP not supported"

        m = osqp.OSQP()

        l = np.copy(self.b)
        l[self.zero:] = -np.inf

        m.setup(
            P = None,
            q = self.c,
            A = self.matrix,
            l = l,
            u = self.b,
            verbose=False,
            eps_abs=1e-16,
            eps_rel=1e-16,
            warm_starting=False,
            polishing=True,
            # verbose=True
            )

        for max_iter in np.linspace(0,10000,101,dtype=int):
            if max_iter > 0:
                m.update_settings(max_iter=max_iter)
                results = m.solve()
                self.x[:] = results.x
                self.y[:] = results.y
            try:
                self.callback_iterate()
            except StopIteration:
                break
        # print(self.solution_qualities)
        # breakpoint()
