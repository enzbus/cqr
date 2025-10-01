# Copyright 2025 Enzo Busseti
#
# This file is part of CQR, the Conic QR Solver.
#
# CQR is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# CQR is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# CQR. If not, see <https://www.gnu.org/licenses/>.
"""Projections on non-symmetric second-order cone.

Plan: we may already use this in a prototype; however before finalizing it
we should figure out which formulation is better in extreme cases (using only
t < 0 case now, may switch), and of course replace scipy.opt call with full
Newton search.
"""

import logging
from unittest import TestCase

import numpy as np
import scipy as sp

logger = logging.getLogger(__name__)

def _base_project_dual_case(t, y, a):
    """Project (t, y) on {(s,z) | s >= ||z*a||_2}, case -||z/a||_2 < t < 0.

    This may be the only nontrivial case we need, because the other one is
    reformulated via dualization into this one.
    """
    assert t < 0
    assert t > -np.linalg.norm(y/a)

    # mu is the Lagrangia multiplier of the equality, s and z depend on it
    # in this case mu is in (-inf, -0.5), s is always positive
    s = lambda mu: t / (1 + 2 * mu)
    z = lambda mu: y / (1 - 2 * mu * a**2)

    # this is the error which we want to be zero
    # error = lambda mu: s(mu) - np.linalg.norm(z(mu) * a)

    # we may gain something with this
    # error = lambda mu: s(mu) - np.linalg.norm(y / (1./a - 2 * mu * a))

    # also with this
    error = lambda mu: -t - np.linalg.norm((1 + 2 * mu) * y / (1./a - 2 * mu * a))

    def error_derivative(mu):
        # redefines z
        denominator = 1.0/a - 2 * mu * a
        z = (1 + 2 * mu) * y / denominator

        # Compute dz/dμ
        v = 1.0/a - 2 * mu * a  # denominator
        dz_dmu = 2 * y * (v + a * (1 + 2 * mu)) / (v ** 2)

        # Compute norm and its derivative
        norm_z = np.linalg.norm(z)

        # d(‖z‖)/dμ = (z/‖z‖) · (dz/dμ)
        dnorm_dmu = np.dot(z / norm_z, dz_dmu)

        # Final derivative: d(error)/dμ = - d(‖z‖)/dμ
        return -dnorm_dmu

    # once replaced with full Newton search we may not even need the bracketing

    # find brackets for the search; we want error(high)>0 and error(low)<0
    # the error is positive for mu close enough to -.5

    # initialize with this
    high = -.5
    low = -1

    # in this case we need to move high a little bit to the left
    if error(high) < 0:
        for _ in range(100):
            # in this case we need to adjust high down from -0.5
            test = (high + low) / 2.
            if error(test) < 0:
                low = test
            else:
                logger.info("First bracket search required %d iters", _)
                high = test
                break
        else:
            raise ValueError("Bracket search failed!")
    else:
        # in this case we need to move low possibly much to the left
        for _ in range(100):
            if error(low) > 0: # the error is positive if mu close to -.5
                high = low
                low *= 2.
            else:
                logger.info("Second bracket search required %d iters", _)
                break
        else:
            raise ValueError("Bracket search failed!")
    logger.info("brackets: (%s, %s)", low, high)

    # replace this with a Newton search
    result = sp.optimize.root_scalar(
        error, x0=(high + low)/2., bracket=(low, high),
        fprime=error_derivative,
        # rtol is around the minimum allowed by impl
        xtol=np.finfo(float).eps, rtol=4*np.finfo(float).eps)
    logger.info(result)
    logger.info("error at root: %s", error(result.root))
    projection = np.concatenate([[s(result.root)], z(result.root)])

    # assert projection on cone surface
    assert np.isclose(
        np.linalg.norm(projection[1:]*a),
        projection[0])

    # assert diff on dual cone surface
    orig_vector = np.concatenate([[t], y])
    assert np.isclose(
        np.linalg.norm((projection[1:]-orig_vector[1:])/a),
        projection[0]-orig_vector[0])

    return projection

def project_nonsymm_soc(x: np.array, a: np.array) -> np.array:
    """Project on the non-symmetric second-order cone.

    This is defined as {(t, y) in R x R^n | t >= ||x * a||_2}, with
    scaling vector a in R^n_++ (all strictly positive elements), and the
    multiplication is elementwise. The standard second-order cone has scaling
    equal to all ones.

    This cone is not self dual: its dual has the (elementwise) inverse scaling
    vector. Projection is not as efficient as projection on the standard SOC,
    it requires an iterative procedure, but can be warm-started.
    
    :param x: Input array, size n+1.
    :param a: Scaling vector, size n.

    :returns: Projection, size n+1.
    """
    assert np.all(a > 0)
    t = x[0]
    y = x[1:]

    # case 1: vector is in cone
    if t >= np.linalg.norm(y * a):
        logger.info('case 1')
        return np.copy(x)

    # case 2: negative of vector in dual cone
    if -t >= np.linalg.norm(y/a):
        logger.info('case 2')
        return np.zeros_like(x)

    # case 3: projection is on non-zero surface of (primal) cone
    # this is split in two sub-cases

    # case 3a: -||z/a||_2 < t < 0
    if t < 0:
        logger.info('case 3a')
        return _base_project_dual_case(t, y, a)

    # case 3b: 0 < t < ||z * a||_2; we dualize
    logger.info('case 3b')
    return x + _base_project_dual_case(-t, -y, 1./a)


class TestNonSymmSOC(TestCase):
    """Just test the projection."""

    accuracy = 1e-13 # about the max we achieve with this test
    size_cone = 1000
    num_tests = 1000

    def test_nonsymm_soc(self):
        """Test projection on non-symmetric SOC."""
        for _ in range(self.num_tests):
            np.random.seed(_)
            x = np.random.randn(self.size_cone)
            # make it difficult enough
            a = np.random.uniform(0, 10., self.size_cone-1)**5
            pi = project_nonsymm_soc(x, a)

            # check pi in cone
            self.assertGreaterEqual(
                pi[0] - np.linalg.norm(pi[1:] * a),
                -self.accuracy
                    * max(1., np.linalg.norm(a))
                    * max(1., np.linalg.norm(pi[1:]))
                    )

            # check pi - x in dual cone
            diff = pi - x
            self.assertGreaterEqual(
                diff[0] - np.linalg.norm(diff[1:] / a),
                -self.accuracy
                    * max(1., np.linalg.norm(diff[1:]))
                    * max(1., np.linalg.norm(1./a))
                    )

            # check pi orthogonal to pi - x
            self.assertLess(abs(np.dot(pi, diff)),
                self.accuracy
                    * max(1., np.linalg.norm(pi))
                    * max(1., np.linalg.norm(diff))
                    )


if __name__ == "__main__":
    from unittest import main
    logging.basicConfig(level='INFO')
    main()
