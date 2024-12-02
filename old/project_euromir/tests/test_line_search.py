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
"""Test the translation to C of dcsrch from lbfgs as interfaced in Python."""

import time  # on msw timing function calls <1s does not work
from unittest import TestCase

import numpy as np
import scipy.sparse as sp

import project_euromir as lib

from .test_equilibrate import _make_Q


class TestLineSearch(TestCase):
    """Test the line search procedure."""

    # we use default parameters from lbfgs3
    good_input = {
        'stp': np.array([1.]),
        'f': np.array([1.]),
        'g': np.array([-1.]),
        'ftol': np.array([1e-3]),
        'gtol': np.array([0.9]),
        'xtol': np.array([0.1]),
        'stpmin': np.array([0.]),
        'stpmax': np.array([1000.]), # in lbfgs this is set iteratively...
        'isave': np.zeros(20, dtype=np.int32),
        'dsave': np.zeros(20, dtype=float),
        'start': True,
    }

    def test_errors(self):
        """Test error codes."""

        # ERROR: STP .LT. STPMIN
        my_input = dict(self.good_input)
        my_input['stpmin'] = np.array([10.])
        self.assertEqual(lib.dcsrch(**my_input), -1)

        # ERROR: STP .GT. STPMAX
        my_input = dict(self.good_input)
        my_input['stpmax'] = np.array([.5])
        self.assertEqual(lib.dcsrch(**my_input), -2)

        # ERROR: INITIAL G .GE. ZERO
        my_input = dict(self.good_input)
        my_input['g'] = np.array([1.])
        self.assertEqual(lib.dcsrch(**my_input), -3)

        # ERROR: FTOL .LT. ZERO
        my_input = dict(self.good_input)
        my_input['ftol'] = np.array([-1.])
        self.assertEqual(lib.dcsrch(**my_input), -4)

        # ERROR: GTOL .LT. ZERO
        my_input = dict(self.good_input)
        my_input['gtol'] = np.array([-1.])
        self.assertEqual(lib.dcsrch(**my_input), -5)

        # ERROR: XTOL .LT. ZERO
        my_input = dict(self.good_input)
        my_input['xtol'] = np.array([-1.])
        self.assertEqual(lib.dcsrch(**my_input), -6)

        # ERROR: STPMIN .LT. ZERO
        my_input = dict(self.good_input)
        my_input['stpmin'] = np.array([-1.])
        self.assertEqual(lib.dcsrch(**my_input), -7)

        # ERROR: STPMAX .LT. STPMIN
        # can't be triggered, that was wrong in original

        # on good input, should get 1
        self.assertEqual(lib.dcsrch(**self.good_input), 1)

    def test_on_simple_gradient_descent(self):
        """Test on simple gradient descent."""

        def loss_and_grad(point):
            loss = point**2
            grad = 2*point
            return loss, grad

        current_point = 1.
        l, g = loss_and_grad(current_point)
        direction = -g
        sign_direction = np.sign(direction)

        my_input = dict(self.good_input)
        my_input['f'] = np.array([l])
        my_input['g'] = np.array([g * sign_direction])
        my_input['stp'] = np.array([direction * sign_direction])

        result = lib.dcsrch(**my_input)
        self.assertEqual(result, 1)
        my_input['start'] = False

        test_point = current_point + direction
        l, g = loss_and_grad(test_point)
        my_input['f'] = np.array([l])
        my_input['g'] = np.array([g * sign_direction])

        result = lib.dcsrch(**my_input)
        self.assertEqual(result, 1)

        test_point = test_point + my_input['stp'][0]
        l, g = loss_and_grad(test_point)
        my_input['f'] = np.array([l])
        my_input['g'] = np.array([g * sign_direction])

        result = lib.dcsrch(**my_input)
        self.assertEqual(result, 0)


if __name__ == '__main__': # pragma: no cover
    from unittest import main
    main()
