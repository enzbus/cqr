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
"""Solver main function."""

import logging
import time

import numpy as np
import scipy as sp

NOHSDE = True

from project_euromir import equilibrate
from project_euromir.direction_calculator import (CGNewton, DenseNewton,
                                                  LSMRLevenbergMarquardt,
                                                  LSQRLevenbergMarquardt,
                                                  WarmStartedCGNewton,
                                                  nocedal_wright_termination)
from project_euromir.line_searcher import (BacktrackingLineSearcher,
                                           LogSpaceLineSearcher,
                                           ScipyLineSearcher)
from project_euromir.loss_no_hsde import (Dresidual, create_workspace, hessian,
                                          loss_gradient, residual)
from project_euromir.refinement import refine

logger = logging.getLogger(__name__)

def solve(matrix, b, c, zero, nonneg,
        # xy = None, # need to import logic for equilibration
        ):
    "Main function."

    print(
        f'PROBLEM SIZE: m={len(b)}, n={len(c)}, zero={zero},'
        f' nonneg={nonneg}, nnz(matrix)={matrix.nnz}')

    # some shape checking
    n = len(c)
    m = len(b)
    assert matrix.shape == (m, n)
    assert zero + nonneg == m

    # if xy is None: # otherwise need equilibration
    #     xy = np.zeros(n+m)

    # equilibration
    d, e, sigma, rho, matrix_transf, b_transf, c_transf = \
    equilibrate.hsde_ruiz_equilibration(
            matrix, b, c, dimensions={
                'zero': zero, 'nonneg': nonneg, 'second_order': ()},
            max_iters=25)

    workspace = create_workspace(m, n, zero)

    def _local_loss(xy):
        return loss_gradient(
            xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf,
            c=c_transf, workspace=workspace)[0]

    def _local_grad(xy):
        # very important, need to copy the output, to redesign
        return np.copy(loss_gradient(
            xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf,
            c=c_transf, workspace=workspace)[1])

    def _local_hessian(xy):
        return hessian(
            xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf,
            c=c_transf, workspace=workspace)

    def _local_residual(xy):
        return residual(
            xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf,
            c=c_transf)

    def _local_derivative_residual(xy):
        return Dresidual(
            xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf,
            c=c_transf)

    xy = np.zeros(n+m)
    loss_xy = _local_loss(xy)
    grad_xy = _local_grad(xy)

    # line_searcher = LogSpaceLineSearcher(
    #     loss_function=_local_loss,
    #     min_step=1e-12,
    #     #gradient_function=_local_grad,
    #     #c_1=1e-4,
    #     #c_2=0.9,
    #     #maxiter=100,
    #     )

    line_searcher = BacktrackingLineSearcher(
        # Scipy searcher is not stable enough, breaks on numerical errors
        # with small steps
        loss_function=_local_loss,
        max_iters=1000)

    direction_calculator = CGNewton(
        # warm start causes issues if null space changes b/w iterations
        hessian_function=_local_hessian,
        rtol_termination=lambda x, g: min(0.5, np.linalg.norm(g)),
        max_cg_iters=None,
        # regularizer=1e-10, # it seems 1e-10 is best, but it's too sensitive to it :(
        )

    # direction_calculator = LSQRLevenbergMarquardt(
    #     residual_function=_local_residual,
    #     derivative_residual_function=_local_derivative_residual,
    #     )

    # LSMR seems better than LSQR and CG, however need to count matrix evals
    # direction_calculator = LSMRLevenbergMarquardt(
    #     residual_function=_local_residual,
    #     derivative_residual_function=_local_derivative_residual,
    #     # warm_start=True, # also doesn't work with warm start
    #     )

    # direction_calculator = DenseNewton( #WarmStartedCGNewton(
    #     hessian_function=_local_hessian,
    #     #rtol_termination=nocedal_wright_termination,
    #     #max_cg_iters=None,
    #     )

    _start = time.time()

    for newton_iterations in range(1000):

        grad_xy = _local_grad(xy)

        logger.info(
            'Iteration %d, current loss %.2e, current inf norm grad %.2e',
            newton_iterations, loss_xy, np.max(np.abs(grad_xy)))

        if np.linalg.norm(grad_xy)/(n+m) < np.finfo(float).eps:
            logger.info('Converged in %d iterations.', newton_iterations)
            break

        direction = direction_calculator.get_direction(
            current_point=xy,
            current_gradient=grad_xy)

        xy, loss_xy, grad_xy = \
            line_searcher.get_next(current_point=xy,
            current_loss=loss_xy,
            current_gradient=grad_xy, direction=direction)

    else:
        raise FloatingPointError(
            f'Solver did not converge in {newton_iterations} iterations.')

    if loss_xy > np.finfo(float).eps:
        raise NotImplementedError(
            'Loss at convergence is not small enough. '
            'Perhaps the program is not primal or dual feasible. '
            'Certificates not yet implemented.')

    print('Newton-CG loop took %.2e seconds' % (time.time() - _start ))

    # create HSDE variables for refinement
    u = np.empty(n+m+1, dtype=float)
    u[:-1] = xy
    u[-1] = 1.
    v = np.zeros_like(u)
    v[n:-1] = -matrix_transf @ u[:n] + b_transf

    for _ in range(3):
        u, v = refine(
            z=u-v, matrix=matrix_transf, b=b_transf, c=c_transf, zero=zero,
            nonneg=nonneg)

    if u[-1] < 1e-8:
        raise FloatingPointError(
            "Refinement failed, Newton-CG solution wasn't good enough.")

    # Transform back into problem format
    u1, u2, u3 = u[:n], u[n:n+m], u[-1]
    v2, v3 = v[n:n+m], v[-1]

    if v3 > u3:
        raise NotImplementedError('Certificates not yet implemented.')

    # Apply HSDE scaling
    x = u1 / u3
    y = u2 / u3
    s = v2 / u3

    # invert Ruiz scaling, copied from other repo
    x_orig = e * x / sigma
    y_orig = d * y / rho
    s_orig = (s/d) / sigma

    return x_orig, y_orig, s_orig
