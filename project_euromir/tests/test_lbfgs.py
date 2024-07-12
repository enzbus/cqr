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

import time  # on msw timing function calls <1s does not work
from unittest import TestCase

import numpy as np
import scipy as sp

import project_euromir as lib
from project_euromir import lbfgs, lbfgs_multiply


class TestLBFGS(TestCase):
    """Test L-BFGS functions."""

    def test_base_direction(self):
        """Test Python implementation of direction calculation."""

        n = 10  # size
        for m in [0, 1, 2, 5]:  # memory

            np.random.seed(m)
            current_gradient = np.random.randn(n)

            past_grad_diffs = np.random.randn(m, n)
            past_steps = np.random.randn(m, n)

            dense_direction = lbfgs_multiply._lbfgs_multiply_dense(
                current_gradient=current_gradient, past_steps=past_steps,
                past_grad_diffs=past_grad_diffs)

            sparse_direction = lbfgs_multiply.lbfgs_multiply(
                current_gradient=current_gradient, past_steps=past_steps,
                past_grad_diffs=past_grad_diffs)

            self.assertTrue(np.allclose(dense_direction, sparse_direction))

            print(dense_direction)
            print(sparse_direction)

    def test_lbfgs_python(self):
        """Test Python implementation of l-bfgs."""

        np.random.seed(0)
        m = 10
        n = 20
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        def loss_and_gradient_function(x):
            residual = A @ x - b
            loss = np.linalg.norm(residual) ** 2
            gradient = 2 * A.T @ residual
            return loss, gradient

        result = sp.optimize.fmin_l_bfgs_b(
            loss_and_gradient_function, x0=np.zeros(n), #iprint=99
            )
        print(result)

        x = lbfgs.minimize_lbfgs(
            loss_and_gradient_function=loss_and_gradient_function,
            initial_point=np.zeros(n), memory=10, max_iters=100, #c_1=1e-3,
            #c_2=0.9,
            max_ls=20)

        self.assertTrue(np.allclose(A @ x, b))
        self.assertLess(
            np.linalg.norm(A @ x - b), np.linalg.norm(A @ result[0] - b))

    def test_projected_lbfgs_python(self):
        """Projected LS."""

        np.random.seed(0)
        m = 10
        n = 20
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        # minimize ||A @ x - b ||^2
        # s.t.     x >= 0

        def loss_and_gradient_function(x):
            x[:] = np.maximum(x, 0.)
            residual = A @ x - b
            loss = np.linalg.norm(residual) ** 2
            gradient = 2 * A.T @ residual
            # zero out gradient going towards constraint (not away from it)
            inactive_set = (x == 0.) & (gradient > 0.)
            gradient[inactive_set] = 0.
            return loss, gradient, ~inactive_set

        result = sp.optimize.fmin_l_bfgs_b(
            loss_and_gradient_function, x0=np.zeros(n), bounds=[[0, None]]*n,
            #iprint=99
            )
        print(result)

        x = lbfgs.minimize_lbfgs(
            loss_and_gradient_function=loss_and_gradient_function,
            initial_point=np.zeros(n), memory=5, max_iters=100, #c_1=1e-3,
            #c_2=0.9,
            max_ls=20, use_active_set=True)

        self.assertGreaterEqual(np.min(x), 0.)
        self.assertLess(
            np.linalg.norm(A @ x - b), np.linalg.norm(A @ result[0] - b))


if __name__ == '__main__':  # pragma: no cover
    import logging
    from unittest import main
    logging.basicConfig(level='INFO')
    main()
