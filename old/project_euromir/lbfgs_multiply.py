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
"""Pure Python implementation of L-BFGS for testing.

Multiplication of the gradient at the current point by the
approximate inverse of the second derivative.

References:

- Original paper:
    Updating quasi-Newton matrices with limited storage, Nocedal 1980
    https://doi.org/10.1090/S0025-5718-1980-0572855-7
    (easy to find non-paywalled)

- Wikipedia page:
    https://en.wikipedia.org/wiki/Limited-memory_BFGS,
    https://web.archive.org/web/20240515120721/https://en.wikipedia.org/wiki/Limited-memory_BFGS

- Blog post:
    https://aria42.com/blog/2014/12/understanding-lbfgs
    https://web.archive.org/web/20231002054213/https://aria42.com/blog/2014/12/understanding-lbfgs
"""

from __future__ import annotations

import logging

import numpy as np
import scipy as sp

logger = logging.getLogger(__name__)

NORMALIZE = False # not sure this helps, and complicates


def lbfgs_multiply(
    current_gradient: np.array,
    past_steps: np.array,
    past_grad_diffs: np.array,
    scale: float | np.array = 1.,
    hessian_approximator = None,
    hessian_cg_iters = None,
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
    :param scale: Diagonal of :math:`H_0`, the base inverse
        Hessian, before the L-BFGS corrections. By default 1, meaning we
        take the identity as base.
    :type scale: float or np.array (1-dimensional)
    """

    memory = past_steps.shape[0]
    assert past_grad_diffs.shape[0] == memory
    assert past_grad_diffs.shape[1] == len(current_gradient)
    assert past_steps.shape[1] == len(current_gradient)

    logger.info(
        'calling lbfgs_multiply with scale %.2e and memory %s',
        scale, memory)

    if NORMALIZE:
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

    else:
        rhos = np.empty(memory, dtype=float)
        for i in range(memory):
            rhos[i] = 1. / np.dot(past_steps[i], past_grad_diffs[i])

    # using paper notation
    q = np.copy(current_gradient)

    # right part, backward iteration
    alphas = np.empty(memory, dtype=float)
    for i in range(memory-1, -1, -1):
        if NORMALIZE:
            alphas[i] = rhos_normalized[i] * np.dot(
                past_steps[i]/norms_steps[i], q)
            q -= alphas[i] * (
                past_grad_diffs[i]/norms_grad_diffs[i])
        else:
            alphas[i] = rhos[i] * np.dot(past_steps[i], q)
            q -= alphas[i] * (past_grad_diffs[i])

    # center part
    if hessian_approximator is not None:
        r = sp.sparse.linalg.cg(
            hessian_approximator, q,
            maxiter=hessian_cg_iters if hessian_cg_iters is not None else 20
            )[0]
    else:
        r = scale * q

    # scale correction, see MINPACK-2/vmlm
    # gamma_correction = np.dot(
    #     past_steps[-1], past_grad_diffs[-1]) / np.dot(
    #         past_grad_diffs[-1], past_grad_diffs[-1])
    # r = gamma_correction * (scale * q)

    # left part, forward iteration
    betas = np.empty(memory, dtype=float)
    for i in range(memory):
        if NORMALIZE:
            betas[i] = rhos_normalized[i] * np.dot(
                past_grad_diffs[i] / np.linalg.norm(past_grad_diffs[i]), r)
            r += (
                alphas[i] * ratios[i] - betas[i]) * (
                    past_steps[i] / norms_steps[i])
        else:
            betas[i] = rhos[i] * np.dot(past_grad_diffs[i], r)
            r += (alphas[i] - betas[i]) * past_steps[i]
    return r


def _lbfgs_multiply_dense(
    current_gradient: np.array,
    past_steps: np.array,
    past_grad_diffs: np.array,
    scale: float | np.array = 1.
):
    """Same as above using dense matrix."""

    memory = past_steps.shape[0]
    assert past_grad_diffs.shape[0] == memory
    assert past_grad_diffs.shape[1] == len(current_gradient)
    assert past_steps.shape[1] == len(current_gradient)

    logger.info(
        'calling lbfgs_multiply_dense with scale %.2e and memory %s',
        scale, memory)

    H = np.diag(
        np.ones(len(current_gradient)) * scale
        if np.isscalar(scale) else scale)

    for i in range(memory):
        rho = 1. / np.dot(past_steps[i], past_grad_diffs[i])
        left = np.eye(len(H)) - rho * np.outer(
            past_steps[i], past_grad_diffs[i])
        right = left.T
        H = left @ H @ right + rho * np.outer(past_steps[i], past_steps[i])

    return H @ current_gradient
