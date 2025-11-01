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
"""New implementation of main solver, based on
``benchmark/implementations/new_new_new_cqr.py``; using python3-numba, simple
functions, and C-like structs; idea is to have them exactly equal to the future
C version; so we can test them using the ctypes bridge implemented in the
project_euromir/__init__.py. In this way we can still test variations in the
benchmark module, especially high level logic (like switch to constant
primal-dual scale, ...) and still have a smooth route to production
implementation. Writing the code in this module will be done incrementally
keeping the benchmark module as testing board (import ... from cqr.new_cqr...).
Then we move it to the cqr namespace and re-publish the Python wheels. 
"""

from .solver import *

# not needed for now

# try:
#     import cvxpy as _cp
#     from .cvxpy_interface import CQR
# except ImportError:
#     pass
