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
"""Unit tests runner."""

# pylint: disable=unused-import

import logging
from unittest import main

from .test_direction_calculator import TestDirectionCalculator
from .test_equilibrate import TestEquilibrate
from .test_lbfgs import TestLBFGS
from .test_line_search import TestLineSearch
from .test_line_searcher import TestLineSearcher
from .test_linear_algebra import TestLinearAlgebra
from .test_loss import TestLoss
from .test_solver import TestSolver

if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
