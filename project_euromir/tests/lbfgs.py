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

"""Pure Python implementation of L-BFGS for testing.
 
We only need the multiplication of the gradient at the current point by the
approximate inverse of the second derivative. Line search and choosing next
point are done externally.

References:

- Original paper:
    Updating quasi-Newton matrices with limited storage, Nocedal 1980
    https://doi.org/10.1090/S0025-5718-1980-0572855-7 
    (easy to find non-paywalled)

- Reference implementation: MINPACK-2, vmlm module
    https://ftp.mcs.anl.gov/pub/MINPACK-2/
    https://github.com/jacobwilliams/MINPACK-2/tree/master/vmlm

- Wikipedia page:
    https://en.wikipedia.org/wiki/Limited-memory_BFGS,
    https://web.archive.org/web/20240515120721/https://en.wikipedia.org/wiki/Limited-memory_BFGS

- Blog post:
    https://aria42.com/blog/2014/12/understanding-lbfgs
    https://web.archive.org/web/20231002054213/https://aria42.com/blog/2014/12/understanding-lbfgs
"""

from __future__ import annotations

import numpy as np


def lbfgs_multiply(
    current_gradient: np.array,
    past_steps: np.array,
    past_grad_diffs: np.array,
    base_inverse_diagonal: float | np.array = 1.
):
    r"""Multiply current gradient by the approximate inverse second derivative.

    :param current_gradient:
    :type current_gradient: np.array (1-dimensional)
    :param past_steps: First dimension is L-BFGS memory. Most recent step is
        last row.
    :type past_steps: np.array (2-dimensional)
    :param past_grad_diffs: First dimension is L-BFGS memory. Most recent 
        gradient difference is last row.
    :type past_grad_diffs: np.array (2-dimensional)
    :param base_inverse_diagonal: Diagonal of :math:`H_0`, the base inverse
        Hessian, before the L-BFGS corrections. By default 1, meaning we
        take the identity as base.
    :type base_inverse_diagonal: float or np.array (1-dimensional)
    """

    memory = past_steps.shape[0]
    assert past_grad_diffs.shape[0] == memory
    assert past_grad_diffs.shape[1] == len(current_gradient)
    assert past_steps.shape[1] == len(current_gradient)

    norms_steps = np.linalg.norm(past_steps, axis=1)
    norms_grad_diffs = np.linalg.norm(past_grad_diffs, axis=1)
    assert len(norms_steps) == memory
    ratios = norms_steps / norms_grad_diffs

    # compute rhos;
    rhos_normalized = np.empty(memory, dtype=float)
    for i in range(memory):
        rhos_normalized[i] = 1. / np.dot(
            past_steps[i]/norms_steps[i],
            past_grad_diffs[i]/norms_grad_diffs[i])

    # using paper notation
    q = np.copy(current_gradient)

    # right part, backward iteration
    alphas = np.empty(memory, dtype=float)
    for i in range(memory-1, -1, -1):
        alphas[i] = rhos_normalized[i] * np.dot(
            past_steps[i]/norms_steps[i], q)
        q -= alphas[i] * (
            past_grad_diffs[i]/norms_grad_diffs[i])

    # center part
    r = base_inverse_diagonal * q

    # scale correction, see MINPACK-2/vmlm
    # gamma_correction = np.dot(
    #     past_steps[-1], past_grad_diffs[-1]) / np.dot(
    #         past_grad_diffs[-1], past_grad_diffs[-1])
    # r = gamma_correction * (base_inverse_diagonal * q)

    # left part, forward iteration
    betas = np.empty(memory, dtype=float)
    for i in range(memory):
        betas[i] = rhos_normalized[i] * np.dot(
            past_grad_diffs[i] / np.linalg.norm(past_grad_diffs[i]), r)
        r += (
            alphas[i] * ratios[i] - betas[i]) * (
                past_steps[i] / norms_steps[i])

    return r


def _lbfgs_multiply_dense(
    current_gradient: np.array,
    past_steps: np.array,
    past_grad_diffs: np.array,
    base_inverse_diagonal: float | np.array = 1.
):
    """Same as above using dense matrix."""

    memory = past_steps.shape[0]
    assert past_grad_diffs.shape[0] == memory
    assert past_grad_diffs.shape[1] == len(current_gradient)
    assert past_steps.shape[1] == len(current_gradient)

    H = np.diag(
        np.ones(len(current_gradient)) * base_inverse_diagonal
        if np.isscalar(base_inverse_diagonal) else base_inverse_diagonal)

    for i in range(memory):
        rho = 1. / np.dot(past_steps[i], past_grad_diffs[i])
        left = np.eye(len(H)) - rho * np.outer(
            past_steps[i], past_grad_diffs[i])
        right = left.T
        H = left @ H @ right + rho * np.outer(past_steps[i], past_steps[i])

    return H @ current_gradient


def strong_wolfe(
        current_loss: float,
        current_gradient: np.array,
        direction: np.array,
        step_size: float,
        next_loss: float,
        next_gradient: np.array,
        c_1: float = 1e-3,
        c_2: float = 0.9):
    """Simple line search satisfying Wolfe conditions.

    This is a much simplified approach versus the canonical dcsrch from
    MINPACK. Hopefully it works!

    Idea: start from unit step. If Armijo rule is not satisfied, backtrack.
    Else if curvature condition is not satisfied, make step longer.

    Default c_1 and c_2 are the same as in MINPACK-2/vmlm, from my tests they
    seem like good choices.
    """

    gradient_times_direction = current_gradient @ direction
    assert gradient_times_direction < 0

    armijo = \
        next_loss <= current_loss + c_1 * step_size * gradient_times_direction

    curvature = abs(next_gradient @ direction) <= c_2 * abs(
        gradient_times_direction)

    return armijo, curvature


def minimize_lbfgs(
        loss_and_gradient_function, initial_point, memory=5, max_iters=100,
        c_1=1e-3, c_2=.9, ls_backtrack=.5, ls_forward=1.1, max_ls=20):
    """Minimize function using back-tracked L-BFGS."""

    n = len(initial_point)

    past_steps = np.empty((memory, n), dtype=float)
    past_grad_diffs = np.empty((memory, n), dtype=float)

    current_point = np.empty(n, dtype=float)
    current_gradient = np.empty(n, dtype=float)
    next_point = np.empty(n, dtype=float)
    next_gradient = np.empty(n, dtype=float)
    direction = np.empty(n, dtype=float)

    current_point[:] = initial_point

    # the function can also modify the current_point (projection)
    current_loss, current_gradient[:] = loss_and_gradient_function(
        current_point)

    for i in range(max_iters):

        if np.linalg.norm(current_gradient) < 1e-16:
            print('CONVERGED')
            break

        print('iter', i)
        print('current_loss', current_loss)
        print('current_gradient', current_gradient)
        print('current_gradient norm', np.linalg.norm(current_gradient))

        if i == 0:
            scale = 1/np.linalg.norm(current_gradient)
        elif memory > 0:
            scale = np.dot(past_steps[-1], past_grad_diffs[-1]) / np.dot(
                past_grad_diffs[-1], past_grad_diffs[-1])

        direction[:] = - lbfgs_multiply(
            current_gradient=current_gradient,
            past_steps=past_steps[memory-i:],
            past_grad_diffs=past_grad_diffs[memory-i:],
            base_inverse_diagonal=scale)

        step_size = 1.
        for _ in range(max_ls):
            next_point[:] = current_point + step_size * direction

            # the function can also modify the next_point (projection)
            next_loss, next_gradient[:] = loss_and_gradient_function(
                next_point)

            armijo, curvature = strong_wolfe(
                current_loss=current_loss, current_gradient=current_gradient,
                direction=direction, step_size=step_size, next_loss=next_loss,
                next_gradient=next_gradient, c_1=c_1, c_2=c_2)

            print(
                'step_size', step_size, 'armijo', armijo, 'curvature', curvature)

            if not armijo:
                step_size *= ls_backtrack
                continue

            if not curvature:
                step_size *= ls_forward
                continue

            # both are satisfied
            past_steps[:-1] = past_steps[1:]
            past_grad_diffs[:-1] = past_grad_diffs[1:]
            if memory > 0:
                past_steps[-1] = next_point - current_point
                past_grad_diffs[-1] = next_gradient - current_gradient
            current_point[:] = next_point
            current_loss = next_loss
            current_gradient[:] = next_gradient
            break

        else:
            print('BACKTRACKING FAILED')
            return current_point
