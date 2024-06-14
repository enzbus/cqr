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
"""HSDE residual and derivative operator."""

import numpy as np

from .cones import add_derivative_embedded_cone_project, embedded_cone_project
from .config import NONUMBA
from .linear_algebra import add_hsde_matvec

if not NONUMBA: # pragma: no cover
    import numba as nb

    from .config import CSCR_MATRIX, DIMENSIONS, B, R


def hsde_residual(z, result, tmp, dimensions, matrix, b, c):
    """Compute HSDE residual.

    :param z: Input array.
    :type z: np.array
    :param result: Result array, will be overwritten
    :type result: np.array
    :param tmp: Temporary work array, will be overwritten.
    :type tmp: np.array
    :param dimensions: Tuple describing the problem and cone sizes.
    :type dimensions: tuple
    :param matrix: Tuple representing the problem matrix in CSC form.
    :type matrix: tuple
    :param b: Constraints right-hand-side vector.
    :type b: np.array
    :param c: Cost vector.
    :type c: np.array
    """
    assert len(result) == len(z)
    result[:] = z

    assert len(tmp) == len(z)
    embedded_cone_project(z=z, result=tmp, dimensions=dimensions)
    result[:] -= tmp

    add_hsde_matvec(
        problem_matrix=matrix, b=b, c=c, array=tmp, result=result,
        invert_sign=False)

if not NONUMBA: # pragma: no cover
    hsde_residual = nb.jit(nb.void(
        R[::1], R[::1], R[::1], DIMENSIONS, CSCR_MATRIX, R[::1], R[::1]),
            nopython=True)(hsde_residual)

def hsde_residual_normalized(z, result, tmp, dimensions, matrix, b, c):
    """Compute HSDE normalized residual.

    :param z: Input array.
    :type z: np.array
    :param result: Result array, will be overwritten
    :type result: np.array
    :param tmp: Temporary work array, will be overwritten.
    :type tmp: np.array
    :param dimensions: Tuple describing the problem and cone sizes.
    :type dimensions: tuple
    :param matrix: Tuple representing the problem matrix in CSC form.
    :type matrix: tuple
    :param b: Constraints right-hand-side vector.
    :type b: np.array
    :param c: Cost vector.
    :type c: np.array
    """
    hsde_residual(z, result, tmp, dimensions, matrix, b, c)
    result /= np.abs(z[-1])

if not NONUMBA: # pragma: no cover
    hsde_residual_normalized = nb.jit(nb.void(
        R[::1], R[::1], R[::1], DIMENSIONS, CSCR_MATRIX, R[::1], R[::1]),
            nopython=True)(hsde_residual_normalized)

def add_hsde_residual_derivative_matvec( # pylint: disable=too-many-arguments
        dimensions, matrix, b, c, tmp, z, array, result, invert_sign=False):
    """Add to result (or subtract) the quantity DR(z) @ dz.

    :param dimensions: Tuple describing the problem and cone sizes.
    :type dimensions: tuple
    :param matrix: Tuple representing the problem matrix in CSC form.
    :type matrix: tuple
    :param b: Constraints right-hand-side vector.
    :type b: np.array
    :param c: Cost vector.
    :type c: np.array
    :param tmp: Temporary work array, will be overwritten.
    :type tmp: np.array
    :param z: Point at which the derivative is computed.
    :type z: np.array

    :param array: Input array.
    :type array: np.array
    :param result: Result array, will be overwritten
    :type result: np.array
    :param invert_sign: Subtract instead of adding.
    :type invert_sign: bool
    """
    assert len(result) == len(z)
    assert len(array) == len(z)

    if not invert_sign:
        result[:] += array
    else:
        result[:] -= array

    assert len(tmp) == len(z)
    tmp[:] = 0.
    add_derivative_embedded_cone_project(
        dimensions=dimensions, z=z, array=array, result=tmp, invert_sign=False)
    if not invert_sign:
        result[:] -= tmp
    else:
        result[:] += tmp

    add_hsde_matvec(
        problem_matrix=matrix, b=b, c=c, array=tmp, result=result,
        invert_sign=invert_sign)

if not NONUMBA: # pragma: no cover
    add_hsde_residual_derivative_matvec = nb.jit(nb.void(
        DIMENSIONS, CSCR_MATRIX, R[::1], R[::1], R[::1], R[::1], R[::1],
        R[::1], B), nopython=True)(add_hsde_residual_derivative_matvec)

def add_hsde_residual_normalized_derivative_matvec(
         # pylint: disable=too-many-arguments
        dimensions, matrix, b, c, tmp, tmp2, z, normalized_residual, array,
        result, invert_sign=False):
    """Add to result (or subtract) the quantity DR(z) @ dz.

    :param dimensions: Tuple describing the problem and cone sizes.
    :type dimensions: tuple
    :param matrix: Tuple representing the problem matrix in CSC form.
    :type matrix: tuple
    :param b: Constraints right-hand-side vector.
    :type b: np.array
    :param c: Cost vector.
    :type c: np.array
    :param tmp: Temporary work array, will be overwritten.
    :type tmp: np.array
    :param tmp2: Temporary work array, will be overwritten.
    :type tmp2: np.array
    :param z: Point at which the derivative is computed.
    :type z: np.array
    :param normalized_residual: Normalized residual of z.
    :type normalized_residual: np.array

    :param array: Input array.
    :type array: np.array
    :param result: Result array, will be overwritten
    :type result: np.array
    :param invert_sign: Subtract instead of adding.
    :type invert_sign: bool
    """
    assert len(tmp2) == len(z)
    tmp2[:] = normalized_residual
    sign = 1. if z[-1] > 0 else -1
    if invert_sign:
        tmp2 *= sign * array[-1]
    else:
        tmp2 *= -sign * array[-1]
    add_hsde_residual_derivative_matvec(
        dimensions=dimensions, matrix=matrix, b=b, c=c, tmp=tmp, z=z,
        array=array, result=tmp2, invert_sign=invert_sign)
    tmp2 /= np.abs(z[-1])
    result[:] = tmp2

if not NONUMBA: # pragma: no cover
    add_hsde_residual_normalized_derivative_matvec = nb.jit(nb.void(
        DIMENSIONS, CSCR_MATRIX, R[::1], R[::1], R[::1], R[::1], R[::1],
        R[::1], R[::1], R[::1], B), nopython=True)(
            add_hsde_residual_normalized_derivative_matvec)


def add_hsde_residual_derivative_rmatvec( # pylint: disable=too-many-arguments
        dimensions, matrix, b, c, tmp, z, array, result, invert_sign=False):
    """Add to result (or subtract) the quantity DR(z).T @ dz.

    :param dimensions: Tuple describing the problem and cone sizes.
    :type dimensions: tuple
    :param matrix: Tuple representing the problem matrix in CSC form.
    :type matrix: tuple
    :param b: Constraints right-hand-side vector.
    :type b: np.array
    :param c: Cost vector.
    :type c: np.array
    :param tmp: Temporary work array, will be overwritten.
    :type tmp: np.array
    :param z: Point at which the derivative is computed.
    :type z: np.array

    :param array: Input array.
    :type array: np.array
    :param result: Result array, will be overwritten
    :type result: np.array
    :param invert_sign: Subtract instead of adding.
    :type invert_sign: bool
    """
    assert len(result) == len(z)
    assert len(array) == len(z)

    if not invert_sign:
        result[:] += array
    else:
        result[:] -= array

    assert len(tmp) == len(z)

    tmp[:] = -array
    add_hsde_matvec(
        problem_matrix=matrix, b=b, c=c, array=array, result=tmp,
        invert_sign=True)

    add_derivative_embedded_cone_project(
        dimensions=dimensions, z=z, array=tmp, result=result,
        invert_sign=invert_sign)


if not NONUMBA: # pragma: no cover
    add_hsde_residual_derivative_rmatvec = nb.jit(nb.void(
        DIMENSIONS, CSCR_MATRIX, R[::1], R[::1], R[::1], R[::1], R[::1],
        R[::1], B), nopython=True)(add_hsde_residual_derivative_rmatvec)

def add_hsde_residual_normalized_derivative_rmatvec(
         # pylint: disable=too-many-arguments
        dimensions, matrix, b, c, tmp, tmp2, z, normalized_residual, array,
        result, invert_sign=False):
    """Add to result (or subtract) the quantity DR(z).T @ dz.

    :param dimensions: Tuple describing the problem and cone sizes.
    :type dimensions: tuple
    :param matrix: Tuple representing the problem matrix in CSC form.
    :type matrix: tuple
    :param b: Constraints right-hand-side vector.
    :type b: np.array
    :param c: Cost vector.
    :type c: np.array
    :param tmp: Temporary work array, will be overwritten.
    :type tmp: np.array
    :param tmp2: Temporary work array, will be overwritten.
    :type tmp2: np.array
    :param z: Point at which the derivative is computed.
    :type z: np.array
    :param normalized_residual: Normalized residual of z.
    :type normalized_residual: np.array

    :param array: Input array.
    :type array: np.array
    :param result: Result array, will be overwritten
    :type result: np.array
    :param invert_sign: Subtract instead of adding.
    :type invert_sign: bool
    """
    assert len(tmp2) == len(z)
    tmp2[:] = 0.
    sign = 1. if z[-1] > 0 else -1
    if invert_sign:
        tmp2[-1] = sign * (array @ normalized_residual)
    else:
        tmp2[-1] = -sign * (array @ normalized_residual)

    add_hsde_residual_derivative_rmatvec(
        dimensions=dimensions, matrix=matrix, b=b, c=c, tmp=tmp, z=z,
        array=array, result=tmp2, invert_sign=invert_sign)

    tmp2 /= np.abs(z[-1])
    result[:] = tmp2

if not NONUMBA: # pragma: no cover
    add_hsde_residual_normalized_derivative_rmatvec = nb.jit(nb.void(
        DIMENSIONS, CSCR_MATRIX, R[::1], R[::1], R[::1], R[::1], R[::1],
        R[::1], R[::1], R[::1], B), nopython=True)(
            add_hsde_residual_normalized_derivative_rmatvec)
