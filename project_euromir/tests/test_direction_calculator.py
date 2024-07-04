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
"""Test descent direction calculation classes."""

import logging
from unittest import TestCase, main

import numpy as np
import scipy as sp

from project_euromir.direction_calculator import (CGNewton, DenseNewton,
                                                  WarmStartedCGNewton)
from project_euromir.line_searcher import BacktrackingLineSearcher

logger = logging.getLogger(__name__)

class TestDirectionCalculator(TestCase):
    """Test descent direction calculation classes."""

    def test_rosenberg_simple(self):
        """Test on a simple Rosenberg Netwon descent."""

        line_searcher = BacktrackingLineSearcher(
            loss_function=sp.optimize.rosen)
        direction_calculator = DenseNewton(
            hessian_function=sp.optimize.rosen_hess)

        for direction_calculator in [
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
