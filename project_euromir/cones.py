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
"""Convex cones projections and derivatives operators."""

import numpy as np

from .config import NONUMBA

if not NONUMBA: # pragma: no cover
    import numba as nb

    from .config import DIMENSIONS, B, R


def zero_project(result):
    """Project on zero cone.

    :param result: Resulting array.
    :type result: np.array
    """
    result[:] = 0.

if not NONUMBA: # pragma: no cover
    zero_project = nb.jit(nb.void(R[::1]), nopython=True)(zero_project)

# skips z, dz
derivative_zero_project = zero_project

def rn_project(z, result):
    """Project on R^n cone.

    :param z: Input array.
    :type z: np.array
    :param result: Resulting array.
    :type result: np.array
    """
    result[:] = z

if not NONUMBA: # pragma: no cover
    rn_project = nb.jit(nb.void(R[::1], R[::1]), nopython=True)(rn_project)

# skips z
derivative_rn_project = rn_project

def nonneg_project(z, result):
    """Project on non-negative cone.

    :param z: Input array.
    :type z: np.array
    :param result: Resulting array.
    :type result: np.array
    """
    result[:] = np.maximum(z, 0.)

if not NONUMBA: # pragma: no cover
    nonneg_project = nb.jit(
        nb.void(R[::1], R[::1]), nopython=True)(nonneg_project)

if NONUMBA:
    def add_derivative_nonneg_project(z, dz, result, invert_sign=False):
        """Add or subtract derivative of projection on non-negative cone.

        :param z: Point at which the derivative is computed.
        :type z: np.array
        :param dz: Input array.
        :type dz: np.array
        :param result: Resulting array.
        :type result: np.array
        :param invert_sign: Whether to subtract instead of adding.
        :type invert_sign: bool
        """
        if not invert_sign:
            result[z > 0] += dz[z > 0]
        else:
            result[z > 0] -= dz[z > 0]

else: # pragma: no cover
    @nb.jit(nb.void(R[::1], R[::1], R[::1], B), nopython=True)
    def add_derivative_nonneg_project(z, dz, result, invert_sign=False):
        """Add or subtract derivative of projection on non-negative cone.

        :param z: Point at which the derivative is computed.
        :type z: np.array
        :param dz: Input array.
        :type dz: np.array
        :param result: Resulting array.
        :type result: np.array
        :param invert_sign: Whether to subtract instead of adding.
        :type invert_sign: bool
        """
        # pylint: disable=consider-using-enumerate
        if not invert_sign:
            for i in range(len(z)):
                if z[i] > 0:
                    result[i] += dz[i]
        else:
            for i in range(len(z)):
                if z[i] > 0:
                    result[i] -= dz[i]


def second_order_project(z, result):
    """Project on second-order cone.

    :param z: Input array.
    :type z: np.array
    :param result: Resulting array.
    :type result: np.array
    """

    assert len(z) >= 2

    y, t = z[1:], z[0]

    # cache this?
    norm_y = np.linalg.norm(y)

    if norm_y <= t:
        result[:] = z
        return

    if norm_y <= -t:
        result[:] = 0.
        return

    result[0] = 1.
    result[1:] = y / norm_y
    result *= (norm_y + t) / 2.


if not NONUMBA: # pragma: no cover
    second_order_project = nb.jit(
        nb.void(R[::1], R[::1]), nopython=True)(
            second_order_project)

def add_derivative_second_order_project(z, dz, result, invert_sign=False):
    """Add or subtract derivative of projection on second-order cone.

    We follow the derivation in `Solution Refinement at Regular Points of Conic
    Problems <https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf>`_.

    :param z: Point at which the derivative is computed.
    :type z: np.array
    :param dz: Input array.
    :type dz: np.array
    :param result: Resulting array.
    :type result: np.array
    :param invert_sign: Whether to subtract instead of adding.
    :type invert_sign: bool
    """

    assert len(z) >= 2
    assert len(z) == len(dz)
    assert len(z) == len(result)

    x, t = z[1:], z[0]

    norm_x = np.linalg.norm(x)

    if norm_x <= t:
        if not invert_sign:
            result[:] += dz
        else:
            result[:] -= dz
        return

    if norm_x <= -t:
        return

    dx, dt = dz[1:], dz[0]

    if not invert_sign:
        result[0] += dt / 2.
        xtdx = x.T @ dx
        result[0] += xtdx / (2. * norm_x)
        result[1:] += x * ((dt - (xtdx/norm_x) * (t / norm_x))/(2 * norm_x))
        result[1:] += dx * ((t + norm_x) / (2 * norm_x))
    else:
        result[0] -= dt / 2.
        xtdx = x.T @ dx
        result[0] -= xtdx / (2. * norm_x)
        result[1:] -= x * ((dt - (xtdx/norm_x) * (t / norm_x))/(2 * norm_x))
        result[1:] -= dx * ((t + norm_x) / (2 * norm_x))

if not NONUMBA: # pragma: no cover
    add_derivative_second_order_project = nb.jit(
        nb.void(R[::1], R[::1], R[::1], B), nopython=True)(
            add_derivative_second_order_project)

def dual_cone_project(z_cone, result, dimensions):
    r"""Project on dual of original problem cone.

    :param z_cone: Conic part of the HSDE z array.
    :type z_cone: np.array
    :param result: Resulting array.
    :type result: np.array
    :param dimensions: Tuple containing the cone dimensions.
    :type dimensions: tuple
    """

    _, zero, nonneg, second_order = dimensions

    cursor = zero

    # zero cone
    result[:cursor] = z_cone[:cursor]

    # nonneg cone
    nonneg_project(
        z_cone[cursor: cursor + nonneg], result[cursor: cursor + nonneg])

    cursor += nonneg

    # second order cone
    for q in second_order:
        second_order_project(z_cone[cursor:cursor+q], result[cursor:cursor+q])
        cursor += q

    assert cursor == len(result)


if not NONUMBA: # pragma: no cover
    dual_cone_project = nb.jit(
        nb.void(R[::1], R[::1], DIMENSIONS), nopython=True)(
            dual_cone_project)


def embedded_cone_project(z, result, dimensions):
    r"""Project on homogeneous self-dual embedding cone.

    That is:

    .. math::

        \mathbf{R}^n \times \mathcal{K}^\star \times \mathbf{R}_+

    Note that the constraints cone :math:`\mathcal{K}` cone is dualized.

    :param z: Input array.
    :type z: np.array
    :param result: Resulting array.
    :type result: np.array
    :param dimensions: Tuple containing the cone dimensions.
    :type dimensions: tuple
    """

    n = dimensions[0]

    # primal variable
    result[:n] = z[:n]

    # cone variable
    dual_cone_project(z[n:-1], result[n:-1], dimensions)

    # homogeneous variable
    result[-1] = 0. if (z[-1] <= 0) else z[-1]

if not NONUMBA: # pragma: no cover
    embedded_cone_project = nb.jit(
        nb.void(R[::1], R[::1], DIMENSIONS), nopython=True)(
            embedded_cone_project)

def add_derivative_dual_cone_project(
        z_cone, dimensions, array, result, invert_sign=False):
    r"""Add or subtract derivative of proj onto the dual of the problem cone.

    :param z_cone: y-s.
    :type z_cone: np.array
    :param dimensions: Tuple containing the cone dimensions.
    :type dimensions: tuple
    :param array: Input array.
    :type array: np.array
    :param result: Array to which to sum or subtract.
    :type result: np.array
    :param invert_sign: Whether to subtract instead of adding.
    :type invert_sign: bool
    """

    _, zero, nonneg, second_order = dimensions

    m = zero + nonneg + sum(second_order)
    assert len(z_cone) == m
    assert len(array) == m
    assert len(result) == m

    cursor = zero

    # primal variable and zero cone
    if invert_sign:
        result[:cursor] -= array[:cursor]
    else:
        result[:cursor] += array[:cursor]

    add_derivative_nonneg_project(
        z_cone[cursor: cursor + nonneg], array[cursor: cursor + nonneg],
        result[cursor: cursor + nonneg], invert_sign=invert_sign)

    cursor += nonneg

    for q in second_order:
        add_derivative_second_order_project(
            z_cone[cursor:cursor+q], array[cursor:cursor+q],
            result[cursor:cursor+q], invert_sign=invert_sign)
        cursor += q

    assert cursor == m

if not NONUMBA: # pragma: no cover
    add_derivative_dual_cone_project = nb.jit(
        nb.void(R[::1], DIMENSIONS, R[::1], R[::1], B), nopython=True)(
            add_derivative_dual_cone_project)

def add_derivative_embedded_cone_project(
        dimensions, z, array, result, invert_sign=False):
    r"""Add or subtract derivative of projection on HSDE cone.

    :param dimensions: Tuple containing the cone dimensions.
    :type dimensions: tuple
    :param z: Point at which the derivative is computed.
    :type z: np.array
    :param array: Input array.
    :type array: np.array
    :param result: Array to which to sum or subtract.
    :type result: np.array
    :param invert_sign: Whether to subtract instead of adding.
    :type invert_sign: bool
    """

    n = dimensions[0]

    cursor = n

    # primal variable
    if invert_sign:
        result[:cursor] -= array[:cursor]
    else:
        result[:cursor] += array[:cursor]

    add_derivative_dual_cone_project(
        z_cone=z[n:-1], dimensions=dimensions, array=array[n:-1],
        result=result[n:-1], invert_sign=invert_sign)

    if invert_sign:
        result[-1] -= 0. if (z[-1] <= 0) else array[-1]
    else:
        result[-1] += 0. if (z[-1] <= 0) else array[-1]


if not NONUMBA: # pragma: no cover
    add_derivative_embedded_cone_project = nb.jit(
        nb.void(DIMENSIONS, R[::1], R[::1], R[::1], B), nopython=True)(
            add_derivative_embedded_cone_project)
