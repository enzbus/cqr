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
"""Base class and implementations for line search algorithms."""

import logging

import numpy as np
import scipy as sp

logger = logging.getLogger(__name__)

class LineSearchError(Exception):
    """Error in line search."""

class LineSearcher:
    """Base class for line search algorithms using strong Wolfe conditions."""

    @property
    def statistics(self):
        return self._statistics if hasattr(self, '_statistics') else {}

    def __init__(
            self, initial_step = 1., c_1=1e-4, c_2=0.9):
        """Initialize with parameters.

        :param initial_step: Initial step to try. Implementations may change
            this dynamically (e.g., try first the step found at last
            invocation).
        :type initial_step: float
        :param c_1: Parameter for Armijo rule.
        :type c_1: float
        :param c_1: Parameter for strong curvature condition.
        :type c_1: float
        """
        self._initial_step = initial_step
        self._c_1 = c_1
        self._c_2 = c_2

    def get_next(
            self, current_point: np.array, current_loss: float,
            current_gradient: np.array, direction: np.array) -> (
                np.array, float, np.array):
        """Get next point along direction.

        :param current_point: Current point.
        :type current_point: np.array
        :param current_loss: Current loss.
        :type current_loss: float
        :param current_gradient: Current gradient.
        :type current_gradient: np.array
        :param direction: Search direction.
        :type direction: np.array

        :returns: Next point, next loss, next gradient or None.
        :rtype: tuple
        """

    def armijo(
            self,
            current_loss: np.array,
            direction_dot_current_gradient: float,
            step_size: float,
            proposed_loss: np.array):
        """Is Armijo rule satisfied?

        :param current_loss: Loss at current point.
        :type current_loss: float
        :param direction_dot_current_gradient: Search direction dot gradient
            at current point.
        :type direction_dot_current_gradient: float
        :param step_size: Length of the step along direction.
        :type step_size: float
        :param proposed_loss: Loss at proposed point.
        :type proposed_loss: float

        :returns: Is Armijo rule satisfied?
        :rtype: bool
        """
        assert direction_dot_current_gradient < 0

        return proposed_loss <= current_loss + self._c_1 * step_size * (
            direction_dot_current_gradient)

    def strong_curvature(
            self,
            current_gradient: np.array,
            direction: np.array,
            proposed_gradient: np.array):
        """Is the strong curvature condition satisfied?

        :param current_gradient: Gradient at current point.
        :type current_gradient: np.array
        :param direction: Search direction.
        :type direction: np.array
        :param proposed_gradient: Gradient at proposed point.
        :type proposed_gradient: np.array

        :returns: Is the strong curvature condition satisfied?
        :rtype: bool
        """

        return np.abs(direction @ proposed_gradient) <= self._c_2 * np.abs(
            direction @ current_gradient)

class BacktrackingLineSearcher(LineSearcher):
    """Backtracking line search implementation."""

    def __init__(
            self, loss_function, initial_step = 1., c_1=1e-4, backtrack=0.9,
            max_iters=100):
        """Initialize with loss function and parameters.

        :param loss_function: Loss function.
        :type loss_function: callable
        :param initial_step: Initial step to try.
        :type initial_step: float
        :param c_1: Parameter for Armijo rule.
        :type c_1: float
        :param backtrack: Backtracking parameter.
        :type backtrack: float
        """
        self._loss_function = loss_function
        self._backtrack = backtrack
        self._max_iters = max_iters
        self._statistics = {'LOSS_EVALS': 0}
        super().__init__(initial_step=initial_step, c_1=c_1, c_2=None)

    def get_next(
            self, current_point: np.array, current_loss: float,
            current_gradient: np.array, direction: np.array) -> np.array:
        """Get next point along direction.

        :param current_point: Current point.
        :type current_point: np.array
        :param current_loss: Current loss.
        :type current_loss: float
        :param current_gradient: Current gradient.
        :type current_gradient: np.array
        :param direction: Search direction.
        :type direction: np.array

        :returns: Next point.
        :rtype: np.array
        """
        direction_dot_current_gradient = current_gradient @ direction
        assert direction_dot_current_gradient < 0
        step_size = float(self._initial_step)
        proposed_point = np.empty_like(current_point)

        for bt_iters in range(self._max_iters):
            proposed_point[:] = current_point + direction * step_size
            proposed_loss = self._loss_function(proposed_point)
            if self.armijo(
                    current_loss=current_loss,
                    direction_dot_current_gradient=direction_dot_current_gradient,
                    step_size=step_size, proposed_loss=proposed_loss):
                logger.info(
                    '%s: step lenght %.2e, function calls %d, current loss '
                    '%.2e, previous loss %.2e', self.__class__.__name__,
                    step_size, bt_iters+1, proposed_loss, current_loss)
                self._statistics['LOSS_EVALS'] += bt_iters
                return (proposed_point, proposed_loss, None)
            step_size *= self._backtrack
        raise LineSearchError(
            "Armijo rule not satisfied after maximum number of backtracks.")

class LinSpaceLineSearcher(LineSearcher):
    """Very simple lin-spaced grid search."""

    def __init__(
            self, loss_function, max_step=1.5, grid_len=100):
        """Initialize with loss function and parameters."""
        self._loss_function = loss_function
        self._steps = np.linspace(0., max_step, grid_len)

    def get_next(
            self, current_point: np.array, current_loss: float,
            current_gradient: np.array, direction: np.array) -> np.array:
        """Get next point along direction.

        :param current_point: Current point.
        :type current_point: np.array
        :param current_loss: Current loss.
        :type current_loss: float
        :param current_gradient: Current gradient.
        :type current_gradient: np.array
        :param direction: Search direction.
        :type direction: np.array

        :returns: Next point.
        :rtype: np.array
        """
        proposed_point = np.empty_like(current_point)
        losses = []
        for step in self._steps:
            proposed_point[:] = current_point + direction * step
            losses.append(self._loss_function(proposed_point))
        best_step = self._steps[np.argmin(losses)]
        best_loss = np.min(losses)
        proposed_point[:] = current_point + direction * best_step
        logger.info(
            '%s: step lenght %.2e, function calls %d, current loss '
            '%.2e, previous loss %.2e', self.__class__.__name__,
            best_step, len(self._steps), best_loss, current_loss)

        return (proposed_point, best_loss, None)


class LogSpaceLineSearcher(LinSpaceLineSearcher):
    """Very simple log-spaced grid search."""

    def __init__(
            self, loss_function, max_step=1.5, min_step=1e-8, grid_len=100):
        """Initialize with loss function and parameters."""
        self._loss_function = loss_function
        self._steps = np.logspace(
            np.log10(min_step), np.log10(max_step), grid_len)


class ScipyLineSearcher(LineSearcher):
    """Scipy's re-implementation of DCSRCH. It's improved over the original.

    This wraps the public function, but there's also a non-public wrapper
    around the original FORTRAN. However it seems it's less reliable.
    """

    def __init__(
            self, loss_function, gradient_function, c_1=1e-4, c_2=0.9,
            maxiter=10):
        """Initialize with parameters.

        :param loss_function: Loss function.
        :type loss_function: callable
        :param gradient_function: Gradient function.
        :type gradient_function: callable
        :param initial_step: Initial step to try. Implementations may change
            this dynamically (e.g., try first the step found at last
            invocation).
        :type initial_step: float
        :param c_1: Parameter for Armijo rule.
        :type c_1: float
        :param c_1: Parameter for strong curvature condition.
        :type c_1: float
        """
        self._loss_function = loss_function
        self._gradient_function = gradient_function
        self._maxiter = maxiter
        self._old_old_fval = None # we store the previous current_loss
        # it's used by the line search to get the initial step lenght
        super().__init__(initial_step=None, c_1=c_1, c_2=c_2)

    def get_next(
            self, current_point: np.array, current_loss: float,
            current_gradient: np.array, direction: np.array) -> np.array:
        """Get next point along direction.

        :param current_point: Current point.
        :type current_point: np.array
        :param current_loss: Current loss.
        :type current_loss: float
        :param current_gradient: Current gradient.
        :type current_gradient: np.array
        :param direction: Search direction.
        :type direction: np.array

        :returns: Next point.
        :rtype: np.array
        """
        assert current_gradient @ direction < 0 # otherwise Nones are returned

        # Pure Python re-implementation of DCSRCH by Scipy; there's also
        # a version that wraps the original FORTRAN, but seems less stable.
        result = sp.optimize.line_search(
            f=self._loss_function,
            myfprime=self._gradient_function,
            xk=current_point,
            pk=direction,
            gfk=current_gradient,
            old_fval=current_loss,
            old_old_fval=self._old_old_fval,
            c1=self._c_1,
            c2=self._c_2,
            maxiter=self._maxiter)
        step_size = result[0]
        logger.info(
            '%s: step lenght %.2e, function calls %d, gradient calls %d, '
            'current loss %.2e, previous loss %.2e',
            self.__class__.__name__, result[0], result[1], result[2],
            result[3], result[4])
        self._old_old_fval = current_loss
        return (current_point + direction * step_size, result[3], result[5])


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    from scipy.optimize import rosen, rosen_der, rosen_hess

    # direction = np.linalg.lstsq(rosen_hess(x), -rosen_der(x), rcond=None)[0]
    # direction = 1 * np.random.randn(5)
    # if direction @ current_grad > 0:
    #     direction = -direction

    for line_searcher in [
            BacktrackingLineSearcher(rosen), LinSpaceLineSearcher(rosen),
            LogSpaceLineSearcher(rosen),
            ScipyLineSearcher(rosen, rosen_der, c_2=0.9)]:

        np.random.seed(10)
        point = np.random.randn(5)
        loss = rosen(point)
        gradient = rosen_der(point)

        for _ in range(10):
            print('loss current:', loss)
            Hessian = rosen_hess(point)
            gradient = gradient if gradient is not None else rosen_der(point)
            point, loss, gradient = line_searcher.get_next(
                current_point=point, current_loss=loss,
                current_gradient=gradient,
                direction=np.linalg.lstsq(
                    rosen_hess(point), -gradient, rcond=None)[0])
            print('loss post:', loss)
