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
"""Base class and implementations for descent direction calculation."""

import logging

import numpy as np
import scipy as sp

logger = logging.getLogger(__name__)

def _densify(linear_operator):  # TODO: move in utils.py
    """Create Numpy 2-d array from a sparse LinearOperator."""
    assert linear_operator.shape[0] == linear_operator.shape[1]
    result = np.eye(linear_operator.shape[0], dtype=float)
    for i in range(len(result)):
        result[:, i] = linear_operator.matvec(result[:, i])
    return result

def nocedal_wright_termination(_current_point, current_gradient):
    """Nocedal-Wright CG rtol termination rule.

    :param current_gradient: Gradient of the loss function at current
        point.
    :type current_gradient: np.array

    :returns: Rtol termination.
    :rtype: float
    """
    return min(0.5, np.sqrt(np.linalg.norm(current_gradient)))

class DirectionCalculator:
    """Base class for descent direction calculation."""

    def get_direction(
        self, current_point: np.array, current_gradient: np.array) -> np.array:
        """Calculate descent direction at current point given current gradient.

        :param current_point: Current point.
        :type current_point: np.array
        :param current_gradient: Gradient of the loss function at current
            point.
        :type current_gradient: np.array

        :returns: Descent direction.
        :rtype: np.array
        """
        raise NotImplementedError

class GradientDescent(DirectionCalculator):
    """Simple gradient descent."""

    def get_direction(
        self, current_point: np.array, current_gradient: np.array) -> np.array:
        """Calculate descent direction at current point given current gradient.

        :param current_point: Current point.
        :type current_point: np.array
        :param current_gradient: Gradient of the loss function at current
            point.
        :type current_gradient: np.array

        :returns: Descent direction.
        :rtype: np.array
        """
        return -current_gradient

class DenseNewton(DirectionCalculator):
    """Calculate descent direction with linear solve of dense Netwon system."""

    def __init__(self, hessian_function):
        """Initialize with function to calculate Hessian.

        :param hessian_function: Function that returns the Hessian. We support
            Numpy 2-d arrays, Scipy sparse matrices, and Scipy sparse Linear
            Operators (which are densified with a Python for loop).
        :type hessian_function: callable
        """
        self._hessian_function = hessian_function

    def get_direction(
        self, current_point: np.array, current_gradient: np.array) -> np.array:
        """Calculate descent direction at current point given current gradient.

        :param current_point: Current point.
        :type current_point: np.array
        :param current_gradient: Gradient of the loss function at current
            point.
        :type current_gradient: np.array

        :returns: Descent direction.
        :rtype: np.array
        """
        hessian = self._hessian_function(current_point)
        if hasattr(hessian, 'todense'):
            hessian = hessian.todense()
        if isinstance(hessian, sp.sparse.linalg.LinearOperator):
            hessian = _densify(hessian)
        assert isinstance(hessian, np.ndarray)
        return np.linalg.lstsq( # we use least-squares solve because it's
            # robust against slight components of the gradient in the null
            # space of the Hessian; in our cases the gradient is always in the
            # span of the Hessian, but numerical errors can sift through
            hessian, -current_gradient, rcond=None)[0]

class CGNewton(DenseNewton):
    """Calculate descent direction using Newton-CG."""

    _x0 = None # overwritten in derived class(es)

    def __init__(
            self, hessian_function,
            rtol_termination=nocedal_wright_termination, max_cg_iters=None):
        """Initialize with function to calculate Hessian.

        :param hessian_function: Function that returns the Hessian. We support
            Numpy 2-d arrays, Scipy sparse matrices, Scipy sparse Linear
            Operators, ..., as long as the returned object implements
            ``__matmul__``.
        :type hessian_function: callable
        :param rtol_termination: Function that given current point and gradient
            returns a value for the rtol termination of CG. Default
            Nocedal-Wright square root rule.
        :type rtol_termination: callable
        :param max_cg_iters: Optionally, maximum number of allowed CG
            iterations. Default None.
        :type max_cg_iters: int or None
        """
        self._hessian_function = hessian_function
        self._rtol_termination = rtol_termination
        self._max_cg_iters = max_cg_iters
        super().__init__(hessian_function=hessian_function)

    def get_direction(
        self, current_point: np.array, current_gradient: np.array) -> np.array:
        """Calculate descent direction at current point given current gradient.

        :param current_point: Current point.
        :type current_point: np.array
        :param current_gradient: Gradient of the loss function at current
            point.
        :type current_gradient: np.array

        :returns: Descent direction.
        :rtype: np.array
        """
        iteration_counter = 0
        def _counter(_):
            nonlocal iteration_counter
            iteration_counter += 1
        current_hessian = self._hessian_function(current_point)
        result = sp.sparse.linalg.cg(
            A=current_hessian,
            b=-current_gradient,
            x0=self._x0, # this is None in this class
            rtol=self._rtol_termination(current_point, current_gradient),
            callback=_counter,
            maxiter=self._max_cg_iters)[0]
        logger.info(
            '%s calculated direction with residual norm %.2e, while the input'
            ' gradient had norm %.2e, in %d iterations',
            self.__class__.__name__,
            np.linalg.norm(current_hessian @ result + current_gradient),
            np.linalg.norm(current_gradient),
            iteration_counter)
        return result

class WarmStartedCGNewton(CGNewton):
    """Calculate descent direction using warm-started Newton CG.

    If you use this class you **must** instantiate different ones for each
    separate system.
    """

    def get_direction(
        self, current_point: np.array, current_gradient: np.array) -> np.array:
        """Calculate descent direction at current point given current gradient.

        :param current_point: Current point.
        :type current_point: np.array
        :param current_gradient: Gradient of the loss function at current
            point.
        :type current_gradient: np.array

        :returns: Descent direction.
        :rtype: np.array
        """
        if self._x0 is None:
            self._x0 = np.zeros_like(current_point, dtype=float)
        direction = super().get_direction(
            current_point=current_point, current_gradient=current_gradient)
        self._x0 = direction
        return direction
