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
"""Solver main function."""

import logging
import time

import numpy as np
import scipy as sp

NOHSDE = True

from project_euromir import equilibrate
from project_euromir.direction_calculator import (
    CGNewton, DenseNewton, DiagPreconditionedCGNewton,
    ExactDiagPreconditionedCGNewton, LSMRLevenbergMarquardt,
    LSQRLevenbergMarquardt, MinResQLPTest, WarmStartedCGNewton,
    nocedal_wright_termination)
from project_euromir.line_searcher import (BacktrackingLineSearcher,
                                           LogSpaceLineSearcher,
                                           ScipyLineSearcher)
from project_euromir.loss_no_hsde import (Dresidual, create_workspace, hessian,
                                          loss_gradient, residual)
from project_euromir.minamide import (MinamideTest, hessian_x_nogap,
                                      hessian_y_nogap)
from project_euromir.refinement import refine

logger = logging.getLogger(__name__)

QR_PRESOLVE = True

def solve(matrix, b, c, zero, nonneg, soc=(),
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
                'zero': zero, 'nonneg': nonneg, 'second_order': soc},
            max_iters=25)

    if QR_PRESOLVE:
        q, r = np.linalg.qr(
            np.vstack([matrix_transf.todense(), c_transf.reshape((1, n))]))
        matrix_transf = q[:-1].A
        c_transf = q[-1].A1
        sigma_qr = np.linalg.norm(
            b_transf) / np.mean(np.linalg.norm(matrix_transf, axis=1))
        b_transf = b_transf/sigma_qr

    workspace = create_workspace(m, n, zero)

    def _local_loss(xy):
        return loss_gradient(
            xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf,
            c=c_transf, workspace=workspace, soc=soc)[0]

    def _local_grad(xy):
        # very important, need to copy the output, to redesign
        return np.copy(loss_gradient(
            xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf,
            c=c_transf, workspace=workspace, soc=soc)[1])

    def _local_hessian(xy):
        return hessian(
            xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf,
            c=c_transf, workspace=workspace, soc=soc)

    def _local_hessian_x_nogap(x):
        return hessian_x_nogap(
            x, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf)

    def _local_hessian_y_nogap(x):
        return hessian_y_nogap(
            x, m=m, n=n, zero=zero, matrix=matrix_transf)

    def _local_residual(xy):
        return residual(
            xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf,
            c=c_transf, soc=soc)

    def _local_derivative_residual(xy):
        return Dresidual(
            xy, m=m, n=n, zero=zero, matrix=matrix_transf, b=b_transf,
            c=c_transf, soc=soc)

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

    direction_calculator = WarmStartedCGNewton(
        # warm start causes issues if null space changes b/w iterations
        hessian_function=_local_hessian,
        rtol_termination=lambda x, g: min(0.5, np.linalg.norm(g)**0.5),
        max_cg_iters=None,
        minres=False,
        regularizer=1e-8, # it seems 1e-10 is best, but it's too sensitive to it :(
        )

    # doesn't improve, yet; we can make many tests
    # direction_calculator = DiagPreconditionedCGNewton(
    #     matrix=matrix_transf,
    #     b=b_transf,
    #     c=c_transf,
    #     zero=zero,
    #     hessian_function=_local_hessian,
    #     rtol_termination=lambda x, g: min(0.5, np.linalg.norm(g)), #,**2),
    #     max_cg_iters=None,
    # )

    # this one also doesn't seem to improve; must be because my diagonal is
    # already quite well scaled, and/or it interacts with the rank 1 part;
    # makes sense to try diag plus rank 1;
    # direction_calculator = ExactDiagPreconditionedCGNewton(
    direction_calculator  = CGNewton(
        # warm start causes issues if null space changes b/w iterations
        hessian_function=_local_hessian,
        rtol_termination=lambda x, g: min(0.5, np.linalg.norm(g)), #,**2),
        max_cg_iters=None,
        # minres=True, # less stable, needs more stringent termination,
        # and/or better logic to transition to refinement, but it is a bit faster
        # it seems
        # regularizer=1e-8, # it seems 1e-10 is best, but it's too sensitive to it :(
        )

    # breaks on little testing, seems that it triggers my stopping condition (?)
    # on a bad point; some more testing is required; may converge in
    # less hessian evals than CG (like minres)
    # direction_calculator = MinResQLPTest(
    #     hessian_function=_local_hessian,
    #     rtol_termination=lambda x, g: min(0.5, np.linalg.norm(g)),
    #     )

    # direction_calculator = LSQRLevenbergMarquardt(
    #     residual_function=_local_residual,
    #     derivative_residual_function=_local_derivative_residual,
    #     )

    # LSMR seems better than LSQR and CG, however need to count matrix evals
    # direction_calculator = LSMRLevenbergMarquardt(
    #     residual_function=_local_residual,
    #     derivative_residual_function=_local_derivative_residual,
        # warm_start=True, # also doesn't work with warm start
    #     )

    # direction_calculator = DenseNewton( #WarmStartedCGNewton(
    #     hessian_function=_local_hessian,
    #     #rtol_termination=nocedal_wright_termination,
    #     #max_cg_iters=None,
    #     )
    # direction_calculator = MinamideTest(
    #     b=b_transf, c=c_transf, h_x_nogap=_local_hessian_x_nogap,
    #     h_y_nogap=_local_hessian_y_nogap)

    _start = time.time()
    # extra_iters=5

    for newton_iterations in range(1000):

        grad_xy = _local_grad(xy)

        logger.info(
            'Iteration %d, current loss %.2e, current inf norm grad %.2e',
            newton_iterations, loss_xy, np.max(np.abs(grad_xy)))

        if ((np.linalg.norm(grad_xy)/(n+m) < np.finfo(float).eps)) and \
                loss_xy < 1e-24:
                # temporary fix, this termination needs to be softer and smarter
            logger.info('Converged in %d iterations.', newton_iterations)
        #     extra_iters -= 1
        # if extra_iters == 0:
            break

        # if loss_xy < np.finfo(float).eps**2:
        #     logger.info('Converged in %d iterations.', newton_iterations)
        #     break

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
    print('Loss at end of Newton-CG loop: %.2e' % loss_xy)
    print('Newton-CG iterations', newton_iterations)
    print('DirectionCalculator statistics', direction_calculator.statistics)
    print('LineSearcher statistics', line_searcher.statistics)

    # create HSDE variables for refinement
    u = np.empty(n+m+1, dtype=float)
    u[:-1] = xy
    u[-1] = 1.
    v = np.zeros_like(u)
    v[n:-1] = -matrix_transf @ u[:n] + b_transf

    for _ in range(3):
        u, v = refine(
            z=u-v, matrix=matrix_transf, b=b_transf, c=c_transf, zero=zero,
            nonneg=nonneg, soc=soc)

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

    if QR_PRESOLVE:
        x = np.linalg.solve(r, x) * sigma_qr
        s *= sigma_qr

    # invert Ruiz scaling, copied from other repo
    x_orig = e * x / sigma
    y_orig = d * y / rho
    s_orig = (s/d) / sigma

    return x_orig, y_orig, s_orig
