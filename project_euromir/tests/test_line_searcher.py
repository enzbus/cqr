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

if __name__ == '__main__': # pragma: no cover
    from unittest import main
    logging.basicConfig(level='INFO')
    main()
