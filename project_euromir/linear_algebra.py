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
"""Basic linear algebra operations."""

from __future__ import annotations

import numpy as np
import scipy as sp


def Q_matvec(
    matrix: sp.csc_matrix,
    b: np.array,
    c: np.array,
    input: np.array) -> np.array:
    """Python implementation of HSDE's Q matvec."""

    m = len(b)
    n = len(c)
    output = np.zeros(m + n + 1)

    output[:n] += matrix.T @ input[n:n+m]
    output[:n] += c * input[-1]

    output[n:n+m] -= matrix @ input[:n]
    output[n:n+m] += b * input[-1]

    output[-1] -= np.dot(c, input[:n])
    output[-1] -= np.dot(b, input[n:n+m])

    return output
