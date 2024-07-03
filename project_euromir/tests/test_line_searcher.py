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
"""Test line search classes."""

import logging
import time  # on msw timing function calls <1s does not work
from unittest import TestCase

logger = logging.getLogger(__name__)

import numpy as np
import scipy as sp
from scipy.optimize import rosen, rosen_der, rosen_hess

import project_euromir.line_searcher as ls


class TestLineSearcher(TestCase):
    """Test line searcher classes."""

    @staticmethod
    def _newton_direction(point, gradient):
        return sp.sparse.linalg.cg(
            rosen_hess(point), -gradient, maxiter=2)[0]

    @staticmethod
    def _gradient_direction(_, gradient):
        return -gradient

    def test_rosenberg(self):
        """Test on simple iterations on the Rosenberg function."""

        for line_searcher in [
            ls.BacktrackingLineSearcher(rosen), ls.LinSpaceLineSearcher(rosen),
            ls.LogSpaceLineSearcher(rosen),
            ls.ScipyLineSearcher(rosen, rosen_der, c_2=0.9)]:
            logger.info(
                'Testing with LineSearcher %s',
                line_searcher.__class__.__name__)

            for direction in [
                self._newton_direction, self._gradient_direction]:

                logger.info(
                    'Testing with direction calculator %s',
                    direction.__name__)

                with self.subTest(
                        line_searcher=line_searcher.__class__.__name__,
                        direction=direction.__name__):
                    np.random.seed(10)
                    point = np.random.randn(5)
                    loss = rosen(point)
                    gradient = rosen_der(point)

                    for _ in range(20):
                        # print('loss current:', loss)
                        gradient = \
                            gradient if gradient is not None else rosen_der(
                                point)
                        oldloss = loss
                        point, loss, gradient = line_searcher.get_next(
                            current_point=point, current_loss=loss,
                            current_gradient=gradient,
                            direction=direction(point, gradient))
                        self.assertLessEqual(loss, oldloss)

if __name__ == '__main__':
    from unittest import main
    logging.basicConfig(level='INFO')
    main()
