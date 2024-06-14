# Copyright 2024 Enzo Busseti.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for cone operations, including accuracy of projection and D."""

from unittest import TestCase, main

import numpy as np

from solver import cones
from solver.config import NONUMBA


class TestCones(TestCase):
    """Unit tests for cone operations."""

    def test_simple_projections(self):
        """Test simple cones projections."""
        for size in [0, 10, 20,]:
            z = np.random.randn(size)
            result = np.empty(len(z))

            cones.zero_project(result)
            self.assertTrue(np.all(result == 0.))

            cones.rn_project(z, result)
            self.assertTrue(np.all(result == z))

            cones.nonneg_project(z, result)
            self.assertTrue(np.all(result >= 0.))

            self.assertTrue(np.all((result == 0.) == (z < 0.)))

    def test_simple_derivatives(self):
        """Test simple cones derivatives."""
        for size in [0, 10, 20,]:
            z = np.random.randn(size)
            dz = np.random.randn(size)
            result = np.empty(len(z))

            cones.derivative_zero_project(result)
            self.assertTrue(np.all(result == 0.))

            cones.derivative_rn_project(dz, result)
            self.assertTrue(np.all(result == dz))

            result[:] = 0
            cones.add_derivative_nonneg_project(
                z, dz, result, invert_sign=False)
            self.assertTrue(np.all(result[z >= 0.] == dz[z >= 0.]))
            self.assertTrue(np.all((result == 0.) == (z < 0.)))
            result[:] = 0
            cones.add_derivative_nonneg_project(
                z, dz, result, invert_sign=True)
            result *= -1
            self.assertTrue(np.all(result[z >= 0.] == dz[z >= 0.]))
            self.assertTrue(np.all((result == 0.) == (z < 0.)))

    # tests accuracies up to size ~1000 with unit norm randn input
    SECOND_ORDER_ERROR = 5e-16 if NONUMBA else 2E-16
    SECOND_ORDER_ERROR_DUAL = 2E-16
    SECOND_ORDER_ORTHOGONALITY_ERROR = 5E-16

    def test_second_order_projection(self):
        """Test projection on second order cone."""

        for size in range(2, 1000):
            np.random.seed(size)
            z = np.random.randn(size)
            z /= np.linalg.norm(z)
            result = np.empty(len(z))

            # check result is in cone
            cones.second_order_project(z, result)
            self.assertLessEqual(
                np.linalg.norm(result[1:]), result[0]+self.SECOND_ORDER_ERROR)

            # check projection is orthogonal
            delta = result - z
            self.assertLessEqual(
                np.abs(delta @ result), self.SECOND_ORDER_ORTHOGONALITY_ERROR)

            # check projection delta is in cone (it's self-dual)
            delta = result - z
            self.assertLessEqual(
                np.linalg.norm(delta[1:]),
                delta[0]+self.SECOND_ORDER_ERROR_DUAL)

    # tests up to size ~1000, unit norm randn input; this is norm of finite
    # difference minus derivative divided by norm of step in the limit of small
    # step
    SECOND_ORDER_DERIVATIVE_ACCURACY = 2E-8

    def test_second_order_derivative(self):
        """Test derivative of projection on second order cone."""

        for size in range(2, 1000):
            np.random.seed(size)
            z = np.random.randn(size)
            z /= np.linalg.norm(z)
            dz = np.random.randn(size)
            d_proj_z = np.empty(len(z))
            proj_z = np.empty(len(z))
            d_proj_z_test = np.empty(len(z))

            cones.second_order_project(z, proj_z)
            d_proj_z[:] = 0
            cones.add_derivative_second_order_project(
                z, dz, d_proj_z, invert_sign= size % 2 == 0)
            if size % 2 == 0:
                d_proj_z *= -1
            best_accuracy = None
            for _ in range(50): # more than enough, only first ~30 are useful
                cones.second_order_project(z+dz, d_proj_z_test)
                finite_difference = d_proj_z_test-proj_z
                accuracy = np.linalg.norm(
                    finite_difference - d_proj_z) / np.linalg.norm(dz)
                if best_accuracy is None:
                    best_accuracy = accuracy
                else:
                    best_accuracy = min(accuracy, best_accuracy)
                # print(_, accuracy)
                if accuracy < self.SECOND_ORDER_DERIVATIVE_ACCURACY:
                    break
                dz /= 2.
                d_proj_z /= 2.
            else:
                self.fail( # pragma: no cover
                    f"Failed at size {size}. "
                    f"Best accuracy achieved: {best_accuracy}.")

    def _test_single_hsde(self, n, zero, nonneg, second_order):
        """Test one instance of HSDE."""
        size = n + zero + nonneg + sum(second_order) + 1
        z = np.random.randn(size)
        result = np.empty(size)

        cones.embedded_cone_project(
            z, result, (n, zero, nonneg, second_order))

        self.assertTrue(np.all(result[:n+zero] == z[:n+zero]))
        self.assertTrue(
            np.all(result[n+zero:n+zero+nonneg] >= 0.))

        cur = n+zero+nonneg
        for q in second_order:
            self.assertLessEqual(
                np.linalg.norm(result[cur+1:cur+q]),
                result[cur]+self.SECOND_ORDER_ERROR)
            cur += q

        self.assertGreaterEqual(result[-1], 0.)

    def test_hsde_pi(self):
        """Test projection on HSDE cone."""
        np.random.seed(0)
        for n in [1, 5, 7]:
            for zero in [0, 5, 10]:
                for nonneg in [0, 5, 10]:
                    for second_order in [
                        np.array([], dtype=int), np.array([5]),
                                np.array([5, 10, 5])]:
                            self._test_single_hsde(
                                n, zero, nonneg, second_order)

    def _test_single_hsde_derivative(self, n, zero, nonneg, second_order):
        """Test derivative of one instance of HSDE."""
        size = n + zero + nonneg + sum(second_order) + 1
        z = np.random.randn(size)
        dz = np.random.randn(size)
        result = np.zeros(size)
        invert_sign = np.random.randn() > 0

        cones.add_derivative_embedded_cone_project(
            dimensions=(n, zero, nonneg, second_order), z=z, array=dz,
            result=result, invert_sign=invert_sign)
        if invert_sign:
            result *= -1

        self.assertTrue(np.all(result[:n+zero] == dz[:n+zero]))
        if z[-1] > 0:
            self.assertEqual(result[-1], dz[-1])
        else:
            self.assertEqual(result[-1], 0.)

    def test_hsde_derivative(self):
        """Test derivative of projection on HSDE cone."""
        np.random.seed(0)
        for n in [1, 5, 7]:
            for zero in [0, 5, 10]:
                for nonneg in [0, 5, 10]:
                    for second_order in [
                        np.array([], dtype=int), np.array([5]),
                                np.array([5, 10, 5])]:
                            self._test_single_hsde_derivative(
                                n, zero, nonneg, second_order)


if __name__ == '__main__':
    main(warnings='error') # pragma: no cover
