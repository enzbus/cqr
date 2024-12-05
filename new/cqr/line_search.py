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
"""Bisection line-search implementation."""
import numpy as np

class LineSearchFailed(Exception):
    """Failure of line search."""

class LineSearcher:
    def __init__(
        self,
        function,
        init_step = 1.,
        max_initial_scalings = 100,
        max_bisections = 20,
        verbose = False,
    ):
        self._function = function
        self.call_counter = 0
        self.low = 0.
        self.f_low = self.function(self.low)
        self.max_initial_scalings = max_initial_scalings
        self.max_bisections = max_bisections
        self.verbose = verbose
        self.init_step = init_step

    def initial_scaling(self):

        f_test = self.function(self.init_step)

        if f_test < self.f_low:
            self.mid = self.init_step
            self.f_mid = f_test
            self.high = np.nan
            self.f_high = np.nan
            self.remove_high_nan()
        else:
            # here we could also be in the case f_high ~ f_low and we are at
            # convergence
            self.mid = np.nan
            self.f_mid = np.nan
            self.high = self.init_step
            self.f_high = f_test
            self.remove_mid_nan()

    def bisection_search(self):

        for _ in range(self.max_bisections):
            self.propagate()
            if self.verbose:
                print()
                print(self)
                print()
            else:
                self.state_valid()
            if max(np.abs(self.f_mid - self.f_low), np.abs(self.f_mid - self.f_high)) < np.finfo(float).eps:
                # converged
                break

    def function(self, step):
        self.call_counter += 1
        return self._function(step)

    def remove_mid_nan(self):
        """Get out of the mid=nan state."""
        for _ in range(self.max_initial_scalings):
            if self.verbose:
                print()
                print(self)
                print()
            else:
                self.state_valid()
            test = (self.low + self.high) / 2.
            f_test = self.function(test)
            # if max(np.abs(f_test - self.f_low), np.abs(f_test - self.f_high)) < np.finfo(float).eps:
            #     raise LineSearchFailed("We are at convergence already.")
            # assert f_test <= self.f_high
            if f_test < self.f_low:
                self.mid = test
                self.f_mid = f_test
                break
            else:
                self.high = test
                self.f_high = f_test
        else:
            raise LineSearchFailed("Could not find small enough step!")

    def remove_high_nan(self):
        """Get out of the high=nan state."""
        for _ in range(self.max_initial_scalings):
            if self.verbose:
                print()
                print(self)
                print()
            else:
                self.state_valid()
            assert self.low == 0.
            test = self.mid * 2.
            f_test = self.function(test)
            # assert f_test <= self.f_low
            if f_test > self.f_mid:
                self.high = test
                self.f_high = f_test
                break
            else:
                self.mid = test
                self.f_mid = f_test
        else:
            raise LineSearchFailed("Could not find long enough step!")

    def propagate(self):
        "Iteration when in normal state."""
        test1 = (self.low + self.mid) / 2.
        test2 = (self.mid + self.high) / 2.
        f_test1 = self.function(test1)
        f_test2 = self.function(test2)
        # print(f(test1), f(test2))

        if f_test1 < self.f_mid:
            self.high = self.mid
            self.f_high = self.f_mid
            self.mid = test1
            self.f_mid = f_test1
            return

        if f_test2 < self.f_mid:
            self.low = self.mid
            self.f_low = self.f_mid
            self.mid = test2
            self.f_mid = f_test2
            return

        self.low = test1
        self.f_low = f_test1
        self.high = test2
        self.f_high = f_test2

    def state_valid(self):
        if not np.isnan(self.mid):
            assert self.low < self.mid
            assert self.f_low >= self.f_mid
        if not np.isnan(self.high):
            assert self.low < self.high
        if not np.isnan(self.mid) and not np.isnan(self.high):
            assert self.mid < self.high
            assert self.f_mid <= self.f_high

    def __repr__(self):
        self.state_valid()

        return f"low={self.low:e},\tmid={self.mid:e},\thigh={self.high:e}\n"\
            + f"f(low)={self.f_low:e},\tf(mid)={self.f_mid:e},\tf(high)={self.f_high:e}"
