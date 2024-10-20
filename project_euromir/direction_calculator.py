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

    @property
    def statistics(self):
        return self._statistics if hasattr(self, '_statistics') else {}

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
    _preconditioner = None # overwritten in derived class(es)

    def __init__(
            self, hessian_function,
            rtol_termination=nocedal_wright_termination, max_cg_iters=None,
            regularizer=0., minres=False, regularizer_callback=None):
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
        self._regularizer = regularizer
        self._minres = minres
        self._regularizer_callback = regularizer_callback
        self._statistics = {'HESSIAN_MATMULS':  0}
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
        if self._regularizer_callback is not None:
            current_hessian = self._hessian_function(
                current_point,
                regularizer=self._regularizer_callback(
                    current_point, current_gradient))
        else:
            current_hessian = self._hessian_function(current_point)
        if self._regularizer > 0.:
            orig_hessian = current_hessian
            current_hessian = sp.sparse.linalg.LinearOperator(
                shape = current_hessian.shape,
                matvec = lambda x: orig_hessian @ x + self._regularizer * x
            )
        # breakpoint()
        # diag = np.diag(self._preconditioner.todense())
        # real_diag = np.diag( _densify(current_hessian))
        # import matplotlib.pyplot as plt
        # plt.plot(diag); plt.plot(real_diag); plt.show()
        # breakpoint()
        result = getattr(sp.sparse.linalg, 'minres' if self._minres else 'cg')(
            A=current_hessian,
            b=-current_gradient,
            x0=self._x0, # this is None in this class
            rtol=self._rtol_termination(current_point, current_gradient),
            M=self._preconditioner, # this is None in this class
            callback=_counter,
            maxiter=self._max_cg_iters)[0]
        self._statistics['HESSIAN_MATMULS'] += iteration_counter
        logger.info(
            '%s calculated direction with residual norm %.2e, while the input'
            ' gradient had norm %.2e, in %d iterations',
            self.__class__.__name__,
            np.linalg.norm(current_hessian @ result + current_gradient),
            np.linalg.norm(current_gradient),
            iteration_counter)
        return result

from .minresQLP import MinresQLP


class MinResQLPTest(DenseNewton):

    def __init__(self, hessian_function, rtol_termination):
        self._rtol_termination = rtol_termination
        # self._max_cg_iters = max_cg_iters
        # self._regularizer = regularizer
        # self._minres = minres
        self._statistics = {'HESSIAN_MATMULS':  0}

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

        current_hessian = self._hessian_function(current_point)
        # breakpoint()
        result = MinresQLP(
            A=current_hessian,
            b=-current_gradient,
            rtol=self._rtol_termination(current_point, current_gradient),
            maxit=1e10)
        # breakpoint()
        print(result[1:])
        self._statistics['HESSIAN_MATMULS'] += result[2]
        return result[0].flatten()

class ExactDiagPreconditionedCGNewton(CGNewton):
    """CG with exact diagonal preconditioning."""

    def get_direction(
        self, current_point: np.array, current_gradient: np.array) -> np.array:

        diag_H = np.array(
            np.diag(_densify(self._hessian_function(current_point))))

        diag_H += 1e-6 #[diag_H < 1e-12] = 1.
        self._preconditioner = sp.sparse.diags(1./diag_H)
        return super().get_direction(
            current_point=current_point, current_gradient=current_gradient)


class DiagPreconditionedCGNewton(CGNewton):
    """CG with diagonal preconditioning."""

    def __init__(self, matrix, b, c, zero, **kwargs):

        self._matrix = matrix
        self._b = b
        m = len(b)
        self._c = c
        n = len(c)
        self._zero = zero
        gap_part = np.concatenate([self._c, self._b])**2
        col_norms = sp.sparse.linalg.norm(matrix, axis=0)**2
        assert len(col_norms) == n
        row_norms = sp.sparse.linalg.norm(matrix, axis=1)**2
        assert len(row_norms) == m
        diag = gap_part
        diag[:n] += col_norms / np.sqrt(2)
        diag[n:] += row_norms
        diag[n+zero:] += 0.5
        self._preconditioner = sp.sparse.diags(1./diag)
        super().__init__(**kwargs)


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

class LSQRLevenbergMarquardt(DirectionCalculator):
    """Calculate descent direction using LSQR-LM."""

    _x0 = None # overwritten in derived class(es)

    def __init__(
            self, residual_function, derivative_residual_function,
            warm_start=False):
        """Initialize with functions to calculate residual and derivative.

        :param residual_function:
        :type residual_function: callable
        :param derivative_residual_function:
        :type derivative_residual_function: callable
        """
        self._residual_function = residual_function
        self._derivative_residual_function = derivative_residual_function
        self._warm_start = warm_start
        self._statistics = {'HESSIAN_MATMULS':  0}

    def _inner_function(self, derivative_residual, residual, current_gradient):
        """In order to replace with LSMR below."""
        result = sp.sparse.linalg.lsqr(
            derivative_residual, -residual, x0=self._x0, calc_var=False,
            atol=min(0.5, np.linalg.norm(current_gradient)), btol=0.)
        return result[0], result[2] # solution, number of iterations

    def get_direction(
        self, current_point: np.array, current_gradient: np.array) -> np.array:
        """Calculate descent direction at current point.

        :param current_point: Current point.
        :type current_point: np.array
        :param current_gradient: Current gradient.
        :type current_gradient: np.array

        :returns: Descent direction.
        :rtype: np.array
        """
        residual = self._residual_function(current_point)
        derivative_residual = self._derivative_residual_function(current_point)
        # breakpoint()
        solution, n_iters = self._inner_function(
            derivative_residual, residual, current_gradient)
        self._statistics['HESSIAN_MATMULS'] += n_iters
        logger.info(
            '%s calculated direction with error norm %.2e, while the input'
            ' gradient had norm %.2e, in %d iterations',
            self.__class__.__name__,
            np.linalg.norm(derivative_residual @ solution + residual),
            np.linalg.norm(current_gradient), n_iters)
        if self._warm_start:
            # LSQR fails with warm-start, somehow gets stuck
            # on directions of very small norm but not good descent
            self._x0 = solution
        return solution

class LSMRLevenbergMarquardt(LSQRLevenbergMarquardt):
    """Calculate descent direction using LSMR-LM."""

    def _inner_function(self, derivative_residual, residual, current_gradient):
        """Just the call to the iterative solver."""
        # breakpoint()
        result = sp.sparse.linalg.lsmr(
            derivative_residual, -residual, x0=self._x0, damp=1e-06, # seems
            # that up to about 1e-6 performance is not affected
            atol=min(0.5, np.linalg.norm(current_gradient)), btol=0.)
        return result[0], result[2] # solution, number of iterations
