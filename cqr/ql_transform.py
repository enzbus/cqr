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
"""Experimental transformation of program data.

In HSDE, with (u1, u2, tau) and (v1, v2, kappa) being the two variables, this
transformation mixes (u1, tau) and (v1, kappa). Conic projection has to keep
that into account, should be relatively easy since transformation is a
triangular matrix. Turns out the cone is unchanged. The u2, v2 variables are
untouched.
"""
import numpy as np
import scipy as sp

def data_ql_transform(A: np.array, b: np.array, c: np.array):
    """Prototype using numpy."""
    n = len(c)
    m = len(b)
    if m < n:
        raise NotImplementedError("Not implemented case n>m yet.")
    matrix = np.block([
        [np.zeros((1, 1)), c.reshape(1, n)],
        [-b.reshape(m, 1), A],
    ])
    _q, _r = np.linalg.qr(matrix[::-1, ::-1])
    q = _q[::-1, ::-1]
    l = _r[::-1, ::-1]
    scale = l[0, 0] # need guard here if 0; when does that happen?
    l /= scale
    q *= scale
    assert np.allclose(
        sp.linalg.solve_triangular(l.T, matrix.T, lower=False), q.T)
    assert np.isclose(l[0, 0], 1.)
    A_transf = q[1:, 1:]
    c_transf = q[0, 1:]
    b_transf = -q[1:, 0]
    return A_transf, c_transf, b_transf, l

def forward_transform_ql(
        u1_init, tau_init, v1_init, kappa_init, n, l):
    """Transform initial guesses into variables for our system."""

    u_tmp = l @ np.concatenate([[tau_init], u1_init])
    v_tmp = sp.linalg.solve_triangular(
        l.T, np.concatenate([[kappa_init], v1_init]), lower=False)
    u1_transf, tau_transf = u_tmp[1:], u_tmp[0]
    v1_transf, kappa_transf = v_tmp[1:], v_tmp[0]

    return u1_transf, tau_transf, v1_transf, kappa_transf

def backward_transform_ql(
        u1_sol, tau_sol, v1_sol, kappa_sol, n, l):
    """Transform solutions back onto original scaling."""

    u_tmp = sp.linalg.solve_triangular(
        l, np.concatenate([[tau_sol], u1_sol]), lower=True)
    v_tmp = l.T @ np.concatenate([[kappa_sol], v1_sol])

    u1_orig, tau_orig = u_tmp[1:], u_tmp[0]
    v1_orig, kappa_orig = v_tmp[1:], v_tmp[0]

    return u1_orig, tau_orig, v1_orig, kappa_orig

if __name__ == '__main__':
    PLOTS = False
    if PLOTS:
        import matplotlib.pyplot as plt
    np.random.seed(0)
    m = 300
    n = 100
    # A = sp.sparse.random(m,n, density=.05).todense().A
    A = np.random.randn(m, n)
    x = np.random.randn(n)
    z = np.random.randn(m)
    y = np.maximum(z, 0.)
    s = y - z
    c = -(A.T @ y)
    b = A @ x + s

    # # QL decomposition
    # newq, newr = np.linalg.qr(A[::-1, ::-1])#, mode='complete')
    # assert np.allclose(A, newq[::-1, ::-1] @ newr[::-1, ::-1])

    # BUILD HSDE_Q (my ordering)
    HSDE_Q_original = np.block([
        [np.zeros((1, 1)), -c.reshape(1, n), -b.reshape(1, m), ],
        [c.reshape(n, 1), np.zeros((n, n)), A.T , ],
        [ b.reshape(m, 1), -A, np.zeros((m, m)),],
    ])

    if PLOTS:
        plt.imshow(HSDE_Q_original)
        plt.colorbar()
        plt.title('HSDE_Q MATRIX BEFORE TRANSFORM')

    A_transf, c_transf, b_transf, l = data_ql_transform(A, b, c)

    # BUILD TRANSFORMED HSDE_Q (my ordering)
    HSDE_Q = np.block([
        [np.zeros((1, 1)), -c_transf.reshape(1, n), -b_transf.reshape(1, m), ],
        [c_transf.reshape(n, 1), np.zeros((n, n)), A_transf.T , ],
        [ b_transf.reshape(m, 1), -A_transf, np.zeros((m, m)),],
    ])

    if PLOTS:
        plt.figure()
        plt.imshow(l)
        plt.colorbar()
        plt.title('L MATRIX')

        plt.figure()
        plt.plot(np.linalg.svd(l)[1])
        plt.title('SINGULAR VALUES OF L MATRIX')

    # CREATE TEST VARIABLES
    u_init = np.random.randn(n+m+1)
    v_init = HSDE_Q_original @ u_init
    tau_init, u1_init, u2_init = u_init[0], u_init[1:1+n], u_init[-m:]
    kappa_init, v1_init, v2_init = v_init[0], v_init[1:1+n], v_init[-m:]

    # FORWARD AND BACKWARD TRANSFORMS
    u1_transf, tau_transf, v1_transf, kappa_transf = forward_transform_ql(
        u1_init, tau_init, v1_init, kappa_init, n, l)
    u1_orig, tau_orig, v1_orig, kappa_orig = backward_transform_ql(
        u1_transf, tau_transf, v1_transf, kappa_transf, n, l)

    # CHECK TRANSFORM INVERTS
    assert np.allclose(u1_init, u1_orig)
    assert np.isclose(tau_init, tau_orig)
    assert np.allclose(v1_init, v1_orig)
    assert np.isclose(kappa_init, kappa_orig)

    ## TEST SCALING
    scaler = sp.linalg.block_diag(np.linalg.inv(l), np.eye(m))
    HSDE_Q_original_scaled = scaler.T @ HSDE_Q_original @ scaler
    assert np.allclose(scaler.T @ HSDE_Q_original @ scaler, HSDE_Q)
    scaler_inverse = sp.linalg.block_diag(l, np.eye(m))
    assert np.allclose(HSDE_Q_original, scaler_inverse.T @ HSDE_Q @ scaler_inverse)

    # TEST TRANSFORM OF VARIABLES CONSISTENT
    # Q_orig @ u_init = v_init
    assert np.allclose(
        scaler.T @ HSDE_Q_original @ scaler @ scaler_inverse @ u_init,
        scaler.T @ v_init)
    assert np.allclose(
        HSDE_Q @ scaler_inverse @ u_init,
        scaler.T @ v_init)
    # u_transf = scaler_inverse @ u_init
    # v_transf = scaler.T @ v_init
    u_transf = np.concatenate([[tau_transf], u1_transf, u2_init])
    v_transf = np.concatenate([[kappa_transf], v1_transf, v2_init])
    assert np.allclose(u_transf, scaler_inverse @ u_init)
    assert np.allclose(v_transf, scaler.T @ v_init)
    assert np.allclose(HSDE_Q @ u_transf, v_transf)

    print("ALL TESTS PASSED.")

    ## STATISTICS
    print('UNIQUE SINGULAR VALUES OF (I + HSDE_Q^T HSDE_Q)')
    print(np.sort(
        np.array(
            list(set(
                np.round(
                    np.linalg.svd(np.eye(1+n+m) + HSDE_Q.T @ HSDE_Q)[1],
                    decimals=6))))))

    if PLOTS:
        plt.figure()
        plt.imshow(HSDE_Q)
        plt.colorbar()
        plt.title('HSDE_Q MATRIX AFTER TRANSFORM')

        plt.figure()
        plt.title('SINGULAR VALUES OF HSDE_Q')
        plt.plot(np.linalg.svd(HSDE_Q)[1], label='transf')
        plt.plot(np.linalg.svd(HSDE_Q_original)[1], label='original')
        plt.legend()
        # plt.show()

        plt.figure()
        plt.title('SINGULAR VALUES OF (I + HSDE_Q^T HSDE_Q)')
        plt.plot(np.linalg.svd(np.eye(1+n+m) + HSDE_Q.T @ HSDE_Q)[1], label='transf')
        plt.plot(np.linalg.svd(np.eye(1+n+m) + HSDE_Q_original.T @ HSDE_Q_original)[1], label='original')
        plt.legend()
        plt.show()
