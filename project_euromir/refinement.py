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
"""Refinement function, simple Python implementation for now."""

import time

import numpy as np
import scipy as sp


def refine(z, matrix, b, c, zero, nonneg):
    """All Python for now, will call all the rest.

    Equilibration is not done here, data must be already transformed.
    """

    # some shape checking
    n = len(c)
    m = len(b)
    assert matrix.shape == (m, n)
    assert zero + nonneg == m

    # temporary, build sparse Q
    Q = sp.sparse.bmat([
        [None, matrix.T, c.reshape(n, 1)],
        [-matrix, None, b.reshape(m, 1)],
        [-c.reshape(1, n), -b.reshape(1, m), None],
        ]).tocsc()

    # define convenience function
    def project(variable):
        result = np.copy(variable)
        result[n+zero:] = np.maximum(result[n+zero:], 0.)
        return result

    # compute u and v by projection
    u = project(z)
    v = u - z

    # very basic, for LPs just compute DR as sparse matrix
    mask = np.ones(len(z), dtype=float)
    mask[n+zero:] = z[n+zero:] > 0
    DR = (Q - sp.sparse.eye(Q.shape[0])) @ sp.sparse.diags(mask
        ) + sp.sparse.eye(Q.shape[0])

    # obtain initial (pre-LSQR) residual
    residual = Q @ u - v
    print('residual norm sq before LSQR', np.linalg.norm(residual)**2)

    # call LSQR
    start = time.time()
    result = sp.sparse.linalg.lsqr(
        DR, residual, atol=0., btol=0., iter_lim=Q.shape[0]*2)
    print('LSQR result[1:-1]', result[1:-1])
    print('LSQR took', time.time() - start)
    dz = result[0]

    # recompute problem variables
    z1 = z - dz
    u = project(z1)
    v = u - z1
    print("residual norm sq after LSQR", np.linalg.norm(Q @ u - v)**2)
    return u, v
