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
"""Test loss, gradient, and Hessian functions."""

import logging
from unittest import TestCase

logger = logging.getLogger(__name__)

import numpy as np
import scipy as sp
from scipy.optimize import check_grad

from project_euromir.direction_calculator import _densify
from project_euromir.loss_no_hsde import (Dresidual, _densify_also_nonsquare,
                                          create_workspace, hessian,
                                          loss_gradient, residual)


class TestLoss(TestCase):
    """Test loss, gradient, and Hessian functions."""

    @classmethod
    def setUpClass(cls):
        """Set up workspace object."""
        # create consts
        np.random.seed(0)
        cls.m = 20
        cls.n = 10
        cls.zero = 5
        cls.nonneg = 15
        cls.matrix = np.random.randn(cls.m, cls.n)
        cls.b = np.random.randn(cls.m)
        cls.c = np.random.randn(cls.n)
        cls.workspace = create_workspace(m=cls.m, n=cls.n, zero=cls.zero)

    @classmethod
    def _loss(cls, xy):
        """Wrapped call to the loss function."""
        return loss_gradient(
            xy, m=cls.m, n=cls.n, zero=cls.zero, matrix=cls.matrix, b=cls.b,
            c=cls.c, workspace=cls.workspace)[0]

    @classmethod
    def _grad(cls, xy):
        """Wrapped call to the gradient function.

        Very important: you need to pass a copy, because the gradient itself
        gets overwritten. Will have to unpack the memory management.
        """
        return np.copy(loss_gradient(
            xy, m=cls.m, n=cls.n, zero=cls.zero, matrix=cls.matrix, b=cls.b,
            c=cls.c, workspace=cls.workspace)[1])

    @classmethod
    def _dense_hessian(cls, xy):
        """Wrapped, and densified, call to the Hessian function."""
        return _densify(
            hessian(
                xy, m=cls.m, n=cls.n, zero=cls.zero, matrix=cls.matrix,
                b=cls.b, c=cls.c, workspace=cls.workspace))

    @classmethod
    def _residual(cls, xy):
        """Wrapped call to the residual function."""
        return np.copy(residual(
            xy, cls.m, cls.n, cls.zero, cls.matrix, cls.b, cls.c))

    @classmethod
    def _dresidual_linop(cls, xy):
        """Wrapped call to the dresidual function, as LinearOperator."""
        return Dresidual(
            xy, cls.m, cls.n, cls.zero, cls.matrix, cls.b, cls.c)

    @classmethod
    def _hessian_from_dresidual(cls, xy):
        dres = cls._dresidual_linop(xy)
        return sp.sparse.linalg.LinearOperator(
            shape=(cls.n+cls.m, cls.n+cls.m),
            matvec=lambda dxy: dres.T @ (dres @ (dxy * 2.)))

    def test_gradient(self):
        """Test that the gradient is numerically accurate."""
        for seed in range(100):
            np.random.seed(seed)
            err = check_grad(
                self._loss, self._grad, np.random.randn(self.n+self.m))
            self.assertLessEqual(err, 6e-5)

    def test_hessian(self):
        """Test that the Hessian is numerically accurate."""
        for seed in range(100):
            np.random.seed(seed)
            err = check_grad(
                self._grad, self._dense_hessian,
                np.random.randn(self.n+self.m))
            self.assertLessEqual(err, 6e-5)

    def test_loss_residual(self):
        """Test that loss and residual functions are consistent."""
        for seed in range(100):
            np.random.seed(seed)
            xy = np.random.randn(self.n+self.m)
            self.assertTrue(
                np.isclose(self._loss(xy),
                    np.linalg.norm(self._residual(xy))**2))

    def test_dr_drt(self):
        """Test that DR and DR^T are consistent."""
        for seed in range(100):
            np.random.seed(seed)
            xy = np.random.randn(self.n+self.m)
            dres = _densify_also_nonsquare(self._dresidual_linop(xy))
            dres_t = _densify_also_nonsquare(self._dresidual_linop(xy).T)
            self.assertTrue(np.allclose(dres.T, dres_t))

    def test_dresidual_gradient(self):
        """Test that (D)residual and gradient are consistent."""
        for seed in range(100):
            np.random.seed(seed)
            xy = np.random.randn(self.n+self.m)
            grad = self._grad(xy)
            newgrad = 2 * (self._dresidual_linop(xy).T @ self._residual(xy))
            self.assertTrue(np.allclose(grad, newgrad))

    def test_dresidual_hessian(self):
        """Test that Dresidual and Hessian are consistent."""
        for seed in range(100):
            np.random.seed(seed)
            xy = np.random.randn(self.n+self.m)
            hess = self._dense_hessian(xy)
            hess_rebuilt = _densify_also_nonsquare(
                self._hessian_from_dresidual(xy))
            self.assertTrue(np.allclose(hess, hess_rebuilt))


if __name__ == '__main__': # pragma: no cover
    from unittest import main
    logging.basicConfig(level='INFO')
    main()
