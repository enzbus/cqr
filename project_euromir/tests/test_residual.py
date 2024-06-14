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
"""Tests for HSDE residual and derivative operator."""


from unittest import TestCase, main

import numpy as np

from solver.cones import dual_cone_project
from solver.config import NONUMBA
from solver.residual import (add_hsde_residual_derivative_matvec,
                             add_hsde_residual_derivative_rmatvec,
                             add_hsde_residual_normalized_derivative_matvec,
                             add_hsde_residual_normalized_derivative_rmatvec,
                             hsde_residual, hsde_residual_normalized)
from solver.solver import Solver
from solver.tests import LSCSTest
from solver.utils import random_problem


class TestResidual(LSCSTest):
    """Unit tests for the HSDE residual."""

    # pylint: disable=protected-access
    def test_residual_in_solver(self):
        """Test correct residual computation, utility function only."""

        matrix, b, c, dimensions = random_problem()
        solver = Solver(*dimensions)
        solver.matrix = matrix
        solver.b = b
        solver.c = c

        solver.x = np.random.randn(solver.n)
        solver.y = np.random.randn(solver.m)
        solver.s = np.random.randn(solver.m)

        for solver.certificate in [True, False]:

            z = solver.z
            residual_numba = np.empty(len(z))
            tmp = np.empty(len(z))
            hsde_residual(z=z, result=residual_numba, tmp=tmp,
                dimensions=solver._dimensions_tuple,
                matrix=solver._matrix_tuple, b=b, c=c)

            # check that it matches textbook computation
            self.assertTrue(np.allclose(residual_numba, solver.hsde_residual))

            # now let's orthogonalize the conic variables independently
            conic_var = solver.y - solver.s
            y = np.empty(solver.m)
            dual_cone_project(conic_var, y, solver._dimensions_tuple)
            solver.y = y
            solver.s = solver.y - conic_var

            self.assertLess(solver.orthogonality_error, 1e-12)
            self.assertLess(solver.primal_cone_error, 1e-12)
            self.assertLess(solver.dual_cone_error, 1e-12)

            # this is the concatenation of the classic residuals
            self.assertTrue(
                np.allclose(residual_numba, solver.concatenated_residuals))

    def test_residual_numba(self):
        """Test numba residual."""

        for i in range(10):
            matrix, b, c, dimensions = random_problem(seed=i)
            solver = Solver(*dimensions)
            solver.matrix = matrix
            solver.b = b
            solver.c = c

            np.random.seed(123*i+456)
            solver.x = np.random.randn(solver.n)
            solver.y = np.random.randn(solver.m)
            solver.s = np.random.randn(solver.m)

            for solver.certificate in [True, False]:

                z = solver.z
                residual_numba = np.empty(len(z))
                tmp = np.empty(len(z))
                hsde_residual(z=z, result=residual_numba, tmp=tmp,
                    dimensions=solver._dimensions_tuple,
                    matrix=solver._matrix_tuple, b=b, c=c)

                # check that it matches textbook computation
                self.assertTrue(
                    np.allclose(residual_numba, solver.hsde_residual))

    def test_normalized_residual(self):
        """Test normalized residual."""
        for i in range(10):
            matrix, b, c, dimensions = random_problem(seed=i)
            solver = Solver(*dimensions)
            solver.matrix = matrix
            solver.b = b
            solver.c = c

            np.random.seed(123*i+456)
            solver.x = np.random.randn(solver.n)
            solver.y = np.random.randn(solver.m)
            solver.s = np.random.randn(solver.m)

            for solver.certificate in [True, False]:

                z = solver.z
                residual_numba = np.empty(len(z))
                tmp = np.empty(len(z))
                hsde_residual_normalized(z=z, result=residual_numba, tmp=tmp,
                    dimensions=solver._dimensions_tuple,
                    matrix=solver._matrix_tuple, b=b, c=c)

                # check that it matches textbook computation
                self.assertTrue(
                    np.allclose(residual_numba,
                    solver.hsde_residual_normalized))

    DR_MATVEC_TOLERANCE = 5e-7

    def _base_test_residual_derivative_matvec(
            self, test_normalized_residual=False):
        """Test matvec the derivative of residual or normalized residual."""

        residual_function = \
            hsde_residual_normalized if test_normalized_residual else \
                hsde_residual

        for i in range(100 if NONUMBA else 200):
            matrix, b, c, dimensions = random_problem(
                n=200, zero=80, nonneg=80, second_order=(
                    40, 40, 40, 40, 40, 40),
                seed=i)
            solver = Solver(*dimensions)
            solver.matrix = matrix
            solver.b = b
            solver.c = c

            np.random.seed(123*i+456)
            solver.x = np.random.randn(solver.n)
            solver.y = np.random.randn(solver.m)
            solver.s = np.random.randn(solver.m)

            for invert_sign in False, True:

                for solver.certificate in [True, False]:

                    z = solver.z
                    # print(z)
                    residual_numba = np.empty(len(z))
                    tmp = np.empty(len(z))
                    residual_function(z=z, result=residual_numba, tmp=tmp,
                        dimensions=solver._dimensions_tuple,
                        matrix=solver._matrix_tuple, b=b, c=c)

                    dz = np.random.randn(len(z))
                    computed_derivative = np.zeros(len(dz))
                    tmp2 = np.empty(len(z))
                    if test_normalized_residual:
                        add_hsde_residual_normalized_derivative_matvec(
                            z=z, array=dz, result=computed_derivative, tmp=tmp,
                            tmp2=tmp2, normalized_residual=residual_numba,
                            dimensions=solver._dimensions_tuple,
                            matrix=solver._matrix_tuple, b=b, c=c,
                            invert_sign=invert_sign)
                    else:
                        add_hsde_residual_derivative_matvec(
                            z=z, array=dz, result=computed_derivative, tmp=tmp,
                            dimensions=solver._dimensions_tuple,
                            matrix=solver._matrix_tuple, b=b, c=c,
                            invert_sign=invert_sign)
                    if invert_sign:
                        computed_derivative = -computed_derivative

                    finite_step = np.empty(len(z))

                    best_accuracy = None
                    for _ in range(50): # only first ~30 seem useful
                        residual_function(z=z+dz, result=finite_step, tmp=tmp,
                            dimensions=solver._dimensions_tuple,
                            matrix=solver._matrix_tuple, b=b, c=c)
                        finite_difference = finite_step-residual_numba
                        accuracy = np.linalg.norm(
                            finite_difference - computed_derivative
                                ) / np.linalg.norm(dz)
                        if best_accuracy is None:
                            best_accuracy = accuracy
                        else:
                            best_accuracy = min(accuracy, best_accuracy)
                        # print(
                        #   i, f'certificate={solver.certificate}', _,
                        #   accuracy)
                        if accuracy < (self.DR_MATVEC_TOLERANCE if not
                            test_normalized_residual
                                else self.DR_MATVEC_TOLERANCE*3):
                            break
                        dz /= 2.
                        computed_derivative /= 2.
                    else: # pragma: no cover
                        self.fail(f"Best accuracy achieved: {best_accuracy}.")

    def test_residual_derivative_matvec(self):
        """Test matvec the derivative of residual."""

        self._base_test_residual_derivative_matvec(
            test_normalized_residual=False)

    def test_normalized_residual_derivative_matvec(self):
        """Test matvec the derivative of residual."""

        self._base_test_residual_derivative_matvec(
            test_normalized_residual=True)

    def test_residual_derivative_rmatvec(self):
        """Test rmatvec the derivative of residual.

        We do it by noting that D (R(z)^2) = 2 DR(z)^T R(z), so we can test
        on the line spanning from z along the direction coming from computed
        DR(z)^T R(z).
        """

        for i in range(100 if NONUMBA else 200):
            matrix, b, c, dimensions = random_problem(
                n=200, zero=80, nonneg=80,
                second_order=(40, 40, 40, 40, 40, 40),
                seed=i)
            solver = Solver(*dimensions)

            solver.matrix = matrix
            solver.b = b
            solver.c = c

            np.random.seed(123*i+456)
            solver.x = np.random.randn(solver.n)
            solver.y = np.random.randn(solver.m)
            solver.s = np.random.randn(solver.m)

            for invert_sign in False, True:

                for solver.certificate in [True, False]:

                    z = solver.z
                    computed_residual = np.empty(len(z))
                    tmp = np.empty(len(z))
                    hsde_residual(z=z, result=computed_residual, tmp=tmp,
                        dimensions=solver._dimensions_tuple,
                        matrix=solver._matrix_tuple, b=b, c=c)

                    residual_square = computed_residual.T @ computed_residual

                    computed_derivative_transpose = np.zeros(len(z))
                    add_hsde_residual_derivative_rmatvec(
                        z=z, array=computed_residual,
                        result=computed_derivative_transpose, tmp=tmp,
                        dimensions=solver._dimensions_tuple,
                        matrix=solver._matrix_tuple, b=b, c=c,
                        invert_sign=invert_sign)
                    if invert_sign:
                        computed_derivative_transpose \
                            = -computed_derivative_transpose

                    finite_step = np.empty(len(z))
                    derivative_transpose_norm = np.linalg.norm(
                        computed_derivative_transpose)

                    best_accuracy = None
                    for _ in range(50): # only first ~30 seem useful
                        hsde_residual(z=z+computed_derivative_transpose,
                            result=finite_step, tmp=tmp,
                            dimensions=solver._dimensions_tuple,
                            matrix=solver._matrix_tuple, b=b, c=c)

                        residual_square_at_finite_step = \
                            finite_step.T @ finite_step

                        step_of_square = (
                            residual_square_at_finite_step - residual_square)
                        from_derivative_transpose = (
                            2 * derivative_transpose_norm *
                            np.linalg.norm(computed_derivative_transpose))

                        # other similar metrics could work too
                        accuracy = step_of_square/from_derivative_transpose - 1
                        # print(_, accuracy)
                        if best_accuracy is None:
                            best_accuracy = accuracy
                        else:
                            best_accuracy = min(accuracy, best_accuracy)
                        if accuracy < 2e-7:
                             break

                        computed_derivative_transpose /= 2.
                    else: # pragma: no cover
                        self.fail(f"Best accuracy achieved: {best_accuracy}.")

    def test_dr_todense(self):
        """Test that the two DR methods match when evaluated to dense."""

        for seed in range(100):
            matrix, b, c, dimensions = random_problem(
                n=5, zero=2, nonneg=2, second_order=(3, 3, 3), density=1.,
                seed=seed)
            solver = Solver(*dimensions)
            solver.matrix = matrix
            np.random.seed(123*seed+456)
            z = np.random.randn(solver.m + solver.n + 1)
            dr_dense = np.zeros((len(z), len(z)))
            drt_dense = np.zeros((len(z), len(z)))
            dz = np.zeros(len(z))
            tmp = np.empty(len(z))

            for i in range(len(z)):
                dz[:] = 0
                dz[i] = 1.
                add_hsde_residual_derivative_matvec(
                            z=z, array=dz,
                            result=dr_dense[i, :], tmp=tmp,
                            dimensions=solver._dimensions_tuple,
                            matrix=solver._matrix_tuple, b=b, c=c,
                            invert_sign=seed % 2)

            for i in range(len(z)):
                dz[:] = 0
                dz[i] = 1.
                add_hsde_residual_derivative_rmatvec(
                            z=z, array=dz,
                            result=drt_dense[i, :], tmp=tmp,
                            dimensions=solver._dimensions_tuple,
                            matrix=solver._matrix_tuple, b=b, c=c,
                            invert_sign=seed % 2)

            self.assertLess(np.linalg.norm(drt_dense - dr_dense.T), 2e-15)

    def test_dr_normalized_todense(self):
        """Test that the two DN methods match when evaluated to dense."""

        for seed in range(100):
            matrix, b, c, dimensions = random_problem(
                n=5, zero=2, nonneg=2, second_order=(3, 3, 3), density=1.,
                seed=seed)
            solver = Solver(*dimensions)
            solver.matrix = matrix
            np.random.seed(123*seed+456)
            z = np.random.randn(solver.m + solver.n + 1)
            dr_dense = np.zeros((len(z), len(z)))
            drt_dense = np.zeros((len(z), len(z)))
            dz = np.zeros(len(z))
            tmp = np.empty(len(z))
            tmp2 = np.empty(len(z))

            normalized_residual = np.empty(len(z))

            hsde_residual_normalized(z=z, result=normalized_residual, tmp=tmp,
                        dimensions=solver._dimensions_tuple,
                        matrix=solver._matrix_tuple, b=b, c=c)

            for i in range(len(z)):
                dz[:] = 0
                dz[i] = 1.
                add_hsde_residual_normalized_derivative_matvec(
                            z=z, array=dz,
                            result=dr_dense[i, :], tmp=tmp, tmp2=tmp2,
                            normalized_residual=normalized_residual,
                            dimensions=solver._dimensions_tuple,
                            matrix=solver._matrix_tuple, b=b, c=c,
                            invert_sign=seed % 2)

            for i in range(len(z)):
                dz[:] = 0
                dz[i] = 1.
                add_hsde_residual_normalized_derivative_rmatvec(
                            z=z, array=dz,
                            result=drt_dense[i, :], tmp=tmp, tmp2=tmp2,
                            normalized_residual=normalized_residual,
                            dimensions=solver._dimensions_tuple,
                            matrix=solver._matrix_tuple, b=b, c=c,
                            invert_sign=seed % 2)

            self.assertAllClose(drt_dense, dr_dense.T)

if __name__ == '__main__':
    main(warnings='error') # pragma: no cover
