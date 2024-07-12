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
"""Test descent direction calculation classes."""

import logging
from unittest import TestCase, main

import numpy as np
import scipy as sp

from project_euromir.direction_calculator import (CGNewton, DenseNewton,
                                                  GradientDescent,
                                                  WarmStartedCGNewton)
from project_euromir.line_searcher import BacktrackingLineSearcher

logger = logging.getLogger(__name__)

class TestDirectionCalculator(TestCase):
    """Test descent direction calculation classes."""

    def test_rosenberg_simple(self):
        """Test on a simple Rosenberg Netwon descent."""

        line_searcher = BacktrackingLineSearcher(
            loss_function=sp.optimize.rosen)

        for direction_calculator in [
            GradientDescent(),
            DenseNewton(hessian_function=sp.optimize.rosen_hess),
            CGNewton(hessian_function=sp.optimize.rosen_hess),
            WarmStartedCGNewton(hessian_function=sp.optimize.rosen_hess)]:

            logger.info("Testing %s", direction_calculator.__class__.__name__)

            np.random.seed(10)
            current_point = np.random.randn(100)
            current_loss = sp.optimize.rosen(current_point)
            current_gradient = None

            for i in range(100):
                if current_gradient is None:
                    current_gradient = sp.optimize.rosen_der(current_point)
                if np.linalg.norm(current_gradient) < np.finfo(float).eps:
                    logger.info('Converged in %d iterations.', i)
                    break
                direction = direction_calculator.get_direction(
                    current_point=current_point,
                    current_gradient=current_gradient)
                # since function is non cvx
                if current_gradient @ direction >= 0: # pragma: no cover
                    logger.info('Bad direction, fallback to steepest descent.')
                    direction = -current_gradient
                current_point, current_loss, current_gradient = \
                    line_searcher.get_next(current_point=current_point,
                    current_loss=current_loss,
                    current_gradient=current_gradient, direction=direction)

if __name__ == '__main__': # pragma: no cover
    logging.basicConfig(level='INFO')
    main()
