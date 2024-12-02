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
"""Pure Python implementation of L-BFGS for testing."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

NORMALIZE = False # not sure this helps, and complicates

from project_euromir import dcsrch

from .lbfgs_multiply import lbfgs_multiply

# TODO plug in interfaced dcsrch function

# input for FORTRAN-to-C-to-Python DCSRCH
DCSRCH_COMMUNICATION = {
    'stp': np.array([1.]),
    'f': np.array([1.]),
    'g': np.array([-1.]),
    'ftol': np.array([1e-3]),
    'gtol': np.array([0.9]),
    'xtol': np.array([1e-8]), # TODO: figure out if this depends on scale
    'stpmin': np.array([0.]),
    'stpmax': np.array([1000.]), # in lbfgsb this is set iteratively...
    'isave': np.zeros(20, dtype=np.int32),
    'dsave': np.zeros(20, dtype=float),
    'start': True,
}

DCSRS_WARNINGS = {
    2: "WARNING: ROUNDING ERRORS PREVENT PROGRESS",
    3: "WARNING: XTOL TEST SATISFIED",
    4: "WARNING: STP = STPMAX",
    5: "WARNING: STP = STPMIN",
    }

# def strong_wolfe(
#         current_loss: float,
#         current_gradient: np.array,
#         direction: np.array,
#         step_size: float,
#         next_loss: float,
#         next_gradient: np.array,
#         c_1: float = 1e-3,
#         c_2: float = 0.9):
#     """Simple line search satisfying Wolfe conditions.

#     This is a much simplified approach versus the canonical dcsrch from
#     MINPACK. Hopefully it works!

#     Idea: start from unit step. If Armijo rule is not satisfied, backtrack.
#     Else if curvature condition is not satisfied, make step longer.

#     Default c_1 and c_2 are the same as in MINPACK-2/vmlm, from my tests they
#     seem like good choices.
#     """

#     gradient_dot_direction = current_gradient @ direction
#     assert gradient_dot_direction < 0

#     armijo = \
#         next_loss <= current_loss + c_1 * step_size * gradient_dot_direction

#     next_gradient_dot_direction = next_gradient @ direction
#     curvature = abs(next_gradient_dot_direction) <= c_2 * abs(
#         gradient_dot_direction)

#     logger.info(
#         'evaluating strong Wolfe conditions with step_size=%s', step_size)
#     logger.info(
#         '\tArmijo is %s, with next_loss=%.2e, current_loss=%.2e'
#         ' gradient_times_direction=%.2e;',
#         armijo, next_loss, current_loss, gradient_dot_direction,
#     )
#     logger.info(
#         '\tcurvature condition is %s with next_gradient_dot_direction=%.2e;',
#         curvature, next_gradient_dot_direction
#     )

#     return armijo, curvature


def minimize_lbfgs(
        loss_and_gradient_function, initial_point, memory=5, max_iters=100,
        c_1=1e-3, c_2=.9, # ls_backtrack=.5, ls_forward=1.1,
        pgtol=1e-6, # same termination as l-bfgs-b
        hessian_approximator=None,
        hessian_cg_iters=20,
        max_ls=20,
        callback=None, use_active_set = False):
    """Minimize function using back-tracked L-BFGS."""

    DCSRCH_COMMUNICATION['ftol'][0] = c_1
    DCSRCH_COMMUNICATION['gtol'][0] = c_2

    func_counter = 0
    n = len(initial_point)

    past_steps = np.empty((memory, n), dtype=float)
    past_grad_diffs = np.empty((memory, n), dtype=float)

    current_point = np.empty(n, dtype=float)
    current_gradient = np.empty(n, dtype=float)
    next_point = np.empty(n, dtype=float)
    next_gradient = np.empty(n, dtype=float)
    direction = np.empty(n, dtype=float)

    current_point[:] = initial_point

    if use_active_set:
        current_active_set = np.empty(n, dtype=bool)
        next_active_set = np.empty(n, dtype=bool)
        # the function can also modify the current_point (projection)
        current_loss, current_gradient[:], current_active_set[:] = loss_and_gradient_function(
            current_point)
    else:
        current_loss, current_gradient[:] = loss_and_gradient_function(
            current_point)
    func_counter += 1

    for i in range(max_iters):

        if (callback is not None) and i > 0:
            callback(current_point)

        if np.max(np.abs(current_gradient)) < pgtol:
            print('PGTOL CONVERGED')
            break

        logger.info('l-bfgs iteration %s: current loss %.17e', i, current_loss)
        if use_active_set:
            logger.info(
                '\tcurrent_active set: %s variables out of %s are active',
                np.sum(current_active_set), len(current_active_set))
        logger.info(
            '\tcurrent_gradient: has norm %s and max abs value %s',
            np.linalg.norm(current_gradient), np.max(np.abs(current_gradient)))

        # print('iter', i)
        # print('current_loss', current_loss)
        # print('current_gradient', current_gradient)
        # print('current_gradient norm', np.linalg.norm(current_gradient))

        # if np.linalg.norm(current_gradient) < np.finfo(float).eps:
        #     print('Converged!')
        #     break

        if i == 0:
            scale = 1/np.linalg.norm(current_gradient)

        elif memory > 0:
            if use_active_set:
                scale = np.dot(past_steps[-1, current_active_set], past_grad_diffs[-1, current_active_set]) / np.dot(
                    past_grad_diffs[-1, current_active_set], past_grad_diffs[-1, current_active_set])
            else:
                scale = np.dot(past_steps[-1], past_grad_diffs[-1]) / np.dot(
                    past_grad_diffs[-1], past_grad_diffs[-1])

        if np.isnan(scale):
            logger.warning('scale calculation resulted in NaN.')
            break

        if use_active_set:
            direction[:] = 0.
            direction[current_active_set] = - lbfgs_multiply(
                current_gradient=current_gradient[current_active_set],
                past_steps=past_steps[max(memory-i, 0):, current_active_set],
                past_grad_diffs=past_grad_diffs[max(memory-i, 0):, current_active_set],
                scale=scale)
            assert np.all(direction[~current_active_set] == 0.)
        else:
            direction[:] = - lbfgs_multiply(
                current_gradient=current_gradient,
                past_steps=past_steps[max(memory-i, 0):],
                past_grad_diffs=past_grad_diffs[max(memory-i, 0):],
                hessian_approximator=hessian_approximator(current_point)
                    if hessian_approximator is not None else None,
                hessian_cg_iters=hessian_cg_iters,
                scale=scale)

        next_loss = current_loss
        next_gradient[:] = current_gradient
        step_size = 1.

        for _ in range(max_ls):
            logger.info('line search iter %s, step size %s', _, step_size)

            # plug in dcsrch
            # breakpoint()
            DCSRCH_COMMUNICATION['start'] = (_ == 0)
            DCSRCH_COMMUNICATION['stp'][0] = step_size
            DCSRCH_COMMUNICATION['f'][0] = next_loss
            DCSRCH_COMMUNICATION['g'][0] = next_gradient @ direction
            # breakpoint()
            dcsrch_result = dcsrch(**DCSRCH_COMMUNICATION)
            if dcsrch_result < 0:
                logger.warning(
                    "Error in line search calling code, are you using active set?")
                print(f'done in iters {i}, tot func calls {func_counter}')
                return next_point
                # breakpoint()

            # breakpoint()

            # line search converged
            if (dcsrch_result == 0) or (dcsrch_result > 1):
                if dcsrch_result != 0:
                    logger.warning(
                        "Line search raised warning %s, exiting.",
                        DCSRS_WARNINGS[dcsrch_result])
                    print(f'done in iters {i}, tot func calls {func_counter}')
                    return next_point

                logger.info('line search converged in %s iterations', _+1)
                past_steps[:-1] = past_steps[1:]
                past_grad_diffs[:-1] = past_grad_diffs[1:]
                if memory > 0:
                    past_steps[-1] = next_point - current_point
                    past_grad_diffs[-1] = next_gradient - current_gradient
                current_point[:] = next_point
                current_loss = next_loss
                current_gradient[:] = next_gradient
                if use_active_set:
                    current_active_set[:] = next_active_set

                break

            step_size =  DCSRCH_COMMUNICATION['stp'][0]

            next_point[:] = current_point + step_size * direction

            if use_active_set:
                # the function can also modify the next_point (projection)
                next_loss, next_gradient[:], next_active_set[:] = \
                    loss_and_gradient_function(next_point)
                logger.info(
                    'next_active_set has %s variables in it that are not'
                    ' in current_active_set, and %s variables not in it that'
                    ' are in current_active_set',
                    np.sum(next_active_set & (~current_active_set)),
                    np.sum(~(next_active_set) & current_active_set),
                )
            else:
                next_loss, next_gradient[:] = loss_and_gradient_function(
                    next_point)
            func_counter += 1
        else:
            logger.warning('line search did not converge')
            break

    print(f'done in iters {i}, tot func calls {func_counter}')
    return current_point

        #     armijo, curvature = strong_wolfe(
        #         current_loss=current_loss, current_gradient=current_gradient,
        #         direction=direction, step_size=step_size, next_loss=next_loss,
        #         next_gradient=next_gradient, c_1=c_1, c_2=c_2)

        #     # print(
        #     #     'step_size', step_size, 'armijo', armijo, 'curvature', curvature)

        #     if not armijo:
        #         step_size *= ls_backtrack
        #         continue

        #     if not curvature:
        #         step_size *= ls_forward
        #         continue

        #     # both are satisfied
        #     logger.info('line search converged in %s iterations', _+1)
        #     past_steps[:-1] = past_steps[1:]
        #     past_grad_diffs[:-1] = past_grad_diffs[1:]
        #     if memory > 0:
        #         past_steps[-1] = next_point - current_point
        #         past_grad_diffs[-1] = next_gradient - current_gradient
        #     current_point[:] = next_point
        #     current_loss = next_loss
        #     current_gradient[:] = next_gradient
        #     if use_active_set:
        #         current_active_set[:] = next_active_set
        #     break

        # else:
        #     print('BACKTRACKING FAILED')
        #     return current_point
