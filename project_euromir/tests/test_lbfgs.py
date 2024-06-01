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

import time  # on msw timing function calls <1s does not work
from unittest import TestCase

import numpy as np
import scipy.sparse as sp

import project_euromir as lib

from . import lbfgs


class TestLBFGS(TestCase):
    """Test L-BFGS functions."""

    def test_base(self):
        """Test Python implementation."""

        n = 10  # size
        for m in [0, 1, 2, 5]:  # memory

            np.random.seed(m)
            current_gradient = np.random.randn(n)

            past_grad_diffs = np.random.randn(m, n)
            past_steps = np.random.randn(m, n)

            dense_direction = lbfgs._lbfgs_multiply_dense(
                current_gradient=current_gradient, past_steps=past_steps,
                past_grad_diffs=past_grad_diffs)

            sparse_direction = lbfgs.lbfgs_multiply(
                current_gradient=current_gradient, past_steps=past_steps,
                past_grad_diffs=past_grad_diffs)

            self.assertTrue(np.allclose(dense_direction, sparse_direction))

            print(dense_direction)
            print(sparse_direction)
