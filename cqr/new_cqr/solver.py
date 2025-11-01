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
"""Solver user interface.

This will stay in Python even when fully implemented in C, or maybe we'll
figure out how to provide a full API using functions? Probably, like
``cqr.factorize(...)``, ``cqr.solve_factorized(...)``,
``cqr.update_factorization(...)``, better than ``cqr.Solver(...)``.
"""

__all__ = []
