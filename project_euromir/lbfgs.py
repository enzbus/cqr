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
"""Pure Python implementation of L-BFGS for testing."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

NORMALIZE = False # not sure this helps, and complicates

from project_euromir import dcsrch

from .lbfgs_multiply import lbfgs_multiply
from .line_search import LineSearchFailed, line_search

# TODO plug in interfaced dcsrch function

# input for FORTRAN-to-C-to-Python DCSRCH
DCSRCH_COMMUNICATION = {
    'stp': np.array([1.]),
    'f': np.array([1.]),
    'g': np.array([-1.]),
    'ftol': np.array([1e-3]),
    'gtol': np.array([0.9]),
    'xtol': np.array([1e-5]), # TODO: figure out if this depends on scale
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
        #c_1=1e-3, c_2=.9, ls_backtrack=.5, ls_forward=1.1,
        pgtol=1e-6, # same termination as l-bfgs-b
        max_ls=20,
        callback=None, use_active_set = False):
    """Minimize function using back-tracked L-BFGS."""

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
                scale=scale)

        current_point_backup = np.copy(current_point)
        current_gradient_backup = np.copy(current_gradient)

        try:

            current_loss = line_search(
                current_point = current_point,
                current_loss = current_loss,
                current_gradient = current_gradient,
                direction = direction,
                loss_gradient_function = loss_and_gradient_function,
                mini_newton_step = 1e-8,
            )

        except LineSearchFailed:
            break

        past_steps[:-1] = past_steps[1:]
        past_grad_diffs[:-1] = past_grad_diffs[1:]
        if memory > 0:
            past_steps[-1] = current_point - current_point_backup
            past_grad_diffs[-1] = current_gradient - current_gradient_backup
        if use_active_set:
            # I don't plan to support AS anyways
            current_loss, current_gradient[:], current_active_set[:] = loss_and_gradient_function(
            current_point)

        continue
        breakpoint()

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
