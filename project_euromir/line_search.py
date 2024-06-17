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
"""Prototype of line search code, to replace dcsrch.

dcsrch is too complicated.

Since the objective function is locally quadratic we can get away with a
simpler scheme. Ideas

- step = 1 does work (wolfe conditions) most of the times, so it's a no brainer
    to try it first

- problems are when it doesn't. then I suggest to do a local approximation of
    the second derivative, by testing two small steps by same amount forward and
    backward. Then we have at least 3 ways to compute the hessian: using the 3
    losses, and the 2 (or even 3) gradient diffs. We can pick as hessian either
    of the these or some average, and bonus we can make the step smaller if
    they don't agree enough. then we do a newton step, and either call it a
    convergence (ignore wolfe) or repeat.

so, I'm proposing a mini-newton on the 1-dimensional problem. from some
research it appears dcsrch has been taken as reference by many, of course
nobody understands what it actually does (by my reading it's some sort of cubic
spline), and i'm betting it's just an overly complicated thing, at least for
our usecase, since we do know the objective is locally quadratic, it's defined
everywhere (minus a measure zero set of no importance), and it's C1.
"""

from __future__ import annotations

import numpy as np

INITIAL_STEP = 1.
WOLFE_C_1 = 1e-3
WOLFE_C_2 = 0.9 # consider tightening it

import logging

logger = logging.getLogger(__name__)

class LineSearchFailed(Exception):
    """Line search failed, we probably reached the numerical limits."""

def strong_wolfe(
        current_loss: float,
        current_gradient_dot_direction: float,
        direction: np.array,
        step_size: float,
        next_loss: float,
        next_gradient: np.array,
        c_1: float = WOLFE_C_1,
        c_2: float = WOLFE_C_2):
    """Simple line search satisfying Wolfe conditions.

    This is a much simplified approach versus the canonical dcsrch from
    MINPACK. Hopefully it works!

    Idea: start from unit step. If Armijo rule is not satisfied, backtrack.
    Else if curvature condition is not satisfied, make step longer.

    Default c_1 and c_2 are the same as in MINPACK-2/vmlm, from my tests they
    seem like good choices.
    """

    armijo = \
        next_loss <= current_loss + c_1 * step_size * current_gradient_dot_direction

    next_gradient_dot_direction = next_gradient @ direction
    curvature = abs(next_gradient_dot_direction) <= c_2 * abs(
        current_gradient_dot_direction)

    logger.info(
        'evaluating strong Wolfe conditions with step_size=%s', step_size)
    logger.info(
        '\tArmijo is %s, with next_loss=%.2e, current_loss=%.2e'
        ' gradient_times_direction=%.2e;',
        armijo, next_loss, current_loss, current_gradient_dot_direction,
    )
    logger.info(
        '\tcurvature condition is %s with next_gradient_dot_direction=%.2e;',
        curvature, next_gradient_dot_direction
    )

    return armijo, curvature


def line_search(
    current_point: np.array,
    current_loss: float,
    current_gradient: np.array,
    direction,
    loss_gradient_function,
    mini_newton_step = 1e-12,
) -> float:

    current_gradient_dot_direction = current_gradient @ direction
    assert current_gradient_dot_direction < 0

    # try step 1
    step_1_loss, step_1_gradient = loss_gradient_function(
            current_point + direction)[:2]

    armijo, curvature = strong_wolfe(
        current_loss = current_loss,
        current_gradient_dot_direction = current_gradient_dot_direction,
        direction = direction,
        step_size = 1,
        next_loss = step_1_loss,
        next_gradient = step_1_gradient)

    if armijo and curvature:
        # exit code
        current_point[:] = current_point + direction
        current_gradient[:] = step_1_gradient
        logger.info('step 1 worked')
        return step_1_loss

    # do local approximations

    for i in range(1):
        # fwd
        mini_step_fwd_loss, mini_step_fwd_gradient = loss_gradient_function(
            current_point + direction * mini_newton_step)[:2]
        mini_step_fwd_gradient_dot_direction = mini_step_fwd_gradient @ direction

        # bwd
        # mini_step_bwd_loss, mini_step_bwd_gradient = loss_gradient_function(
        #     current_point - direction * mini_newton_step)
        # mini_step_bwd_gradient_dot_direction = mini_step_bwd_gradient @ direction

        # compute 3 hessians, we may actually just do the first one
        hessian_1 = (mini_step_fwd_gradient_dot_direction - current_gradient_dot_direction)/mini_newton_step
        # hessian_2 = (current_gradient_dot_direction - mini_step_bwd_gradient_dot_direction)/mini_newton_step
        # hessian_3 = (2 * mini_step_fwd_loss + 2 * mini_step_bwd_loss - current_loss)/mini_newton_step**2

        # do something if they disagree
        # ...

        # breakpoint()

        # if these don't hold, we need smaller steps
        if not (hessian_1 > 0):
            raise LineSearchFailed
            logger.warning('Backtracking the newton search step!')
            mini_newton_step = mini_newton_step/2
        else:
            break
    # else:
    #     raise LineSearchFailed
    logger.info('hessian is %s', hessian_1)
    # assert hessian_2 > 0
    # assert hessian_3 > 0

    # just this?
    # hessian = (hessian_1 + hessian_2 + hessian_3) / 3
    # assert hessian > 0.

    # then, we have a step
    chosen_step = -current_gradient_dot_direction / hessian_1

    for i in range(20):

        chosen_step_loss, chosen_step_gradient = loss_gradient_function(
                current_point + direction * chosen_step)[:2]

        armijo, curvature = strong_wolfe(
            current_loss = current_loss,
            current_gradient_dot_direction = current_gradient_dot_direction,
            direction = direction,
            step_size = chosen_step,
            next_loss = chosen_step_loss,
            next_gradient = chosen_step_gradient)

        if armijo: # and curvature:
            logger.info('wolfe (armijo only) worked with newton step!')

            # exit code
            current_point[:] = current_point + direction * chosen_step
            current_gradient[:] = chosen_step_gradient
            return chosen_step_loss

        logger.info('backtracking the newton step')
        chosen_step *= .5

    logger.warning('line search failed!')
    raise LineSearchFailed

    # exit code
    current_point[:] = current_point + direction * chosen_step
    current_gradient[:] = chosen_step_gradient
    return chosen_step_loss

    # just do this, ignore wolfe
    for i in range(20):
        chosen_step_loss, chosen_step_gradient = loss_gradient_function(
            current_point + direction * chosen_step * 0.5**i
        )
        if chosen_step_loss < current_loss:
            break

    # exit code
    current_point[:] = current_point + direction * chosen_step * 0.5**i
    current_gradient[:] = chosen_step_gradient
    return chosen_step_loss
