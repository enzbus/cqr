# Copyright 2024 Enzo Busseti
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
"""Linear space projection."""

import numpy as np
import scipy as sp


def linspace_project(tau, u1, u2, kappa, v2, orthogonal_matrix, scale):
    """Projection on linear space under QL transform."""

    n = len(u1)
    m = len(u2)

    # use this ordering
    u = np.concatenate([u1, [tau], u2])
    v = np.concatenate([np.zeros(n), [kappa], v2])

    # we'll have to fix the ql transf to produce this ordering
    reordered_orth = np.empty_like(orthogonal_matrix)
    reordered_orth[:, -1] = orthogonal_matrix[:, 0]
    reordered_orth[:, :-1] = orthogonal_matrix[:, 1:]
    # reordered_orth = np.block([
    #     [c.reshape(1, n), corrector],
    #     [A, -b.reshape(m, 1)],
    # ])

    newb = reordered_orth[:, -1]
    newc = reordered_orth[0, :]
    newAc = reordered_orth[:, :-1]
    newAb = reordered_orth[1:, :]
    # breakpoint()

    ##########
    # this is the matrix form we work with
    lower_Q = np.block([
        [np.zeros((n, n+1)), np.zeros((n, m))],
        [-reordered_orth, np.zeros((m+1, m))]
    ]) * scale

    # This is the format, see test below
    Q = lower_Q - lower_Q.T

    ##########
    PLOT = False
    if PLOT:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        im1 = axes[0, 0].imshow(lower_Q @ lower_Q)
        axes[0, 0].set_title('lower_Q @ lower_Q')
        fig.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(lower_Q @ lower_Q.T)
        axes[0, 1].set_title('lower_Q @ lower_Q.T')
        fig.colorbar(im2, ax=axes[0, 1])

        im3 = axes[1, 0].imshow(lower_Q.T @ lower_Q)
        axes[1, 0].set_title('lower_Q.T @ lower_Q')
        fig.colorbar(im3, ax=axes[1, 0])

        im4 = axes[1, 1].imshow(lower_Q.T @ lower_Q.T)
        axes[1, 1].set_title('lower_Q.T @ lower_Q.T')
        fig.colorbar(im4, ax=axes[1, 1])

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        plt.figure()
        Q = lower_Q - lower_Q.T
        plt.imshow(np.eye(n+m+1) + Q.T @ Q)
        plt.colorbar()

        plt.show()

    ##########
    TEST = True
    if TEST:
        # To test
        A = reordered_orth[1:, :-1] * scale
        b = -reordered_orth[1:, -1] * scale
        c = reordered_orth[0, :-1] * scale
        # corrector = orthogonal_matrix[0,0]
        # reordered_orth = np.block([
        #     [c.reshape(1, n), corrector],
        #     [A, -b.reshape(m, 1)],
        # ])
        Q_test = np.block([
            [ np.zeros((n, n)), c.reshape(n, 1), A.T],
            [-c.reshape(1, n), np.zeros((1, 1)),  -b.reshape(1, m)],
            [ -A, b.reshape(m, 1),  np.zeros((m, m))],
        ])
        assert np.allclose(Q_test, lower_Q - lower_Q.T)
    ##########

    ##########
    # basic implementation
    # u = np.concatenate([u1, [tau], u2])
    # v = np.concatenate([np.zeros(n), [kappa], v2])
    # Q = lower_Q - lower_Q.T
    # MAT = np.eye(n+m+1) + Q.T @ Q
    # u_star = np.linalg.solve(MAT, u + Q.T @ v)
    # v_star = Q @ u_star

    # Part 1
    # compute Q.T @ v
    Qtv = scale * (
        - np.concatenate([reordered_orth.T @ np.concatenate([[kappa], v2]), np.zeros(m)]) # lower_Q.T @ v
        + np.concatenate([np.zeros(n), kappa * newb]) # - lower_Q @ v
    )

    # Part 1.5, assemble RHS
    RHS = u + Qtv

    # Part 2, build matrix, using reordered_orth and scale
    # np.eye(n+m+1)
    base_diag = np.ones(n+m+1)
    # lower_Q.T @ lower_Q
    base_diag[:n+1] += scale ** 2
    # lower_Q @ lower_Q.T
    low_right = sp.linalg.block_diag(
        np.zeros((n,n)), reordered_orth @ reordered_orth.T) * scale**2
    # lower_Q @ lower_Q
    low_left = np.block([
        [np.zeros((n, n+1)), np.zeros((n, m))],
        [np.outer(scale * newb, scale * newc), np.zeros((m+1, m))],
    ])
    assert np.allclose(low_left, lower_Q @ lower_Q)
    MAT = (
        + np.diag(base_diag) # np.eye(n+m+1) + lower_Q.T @ lower_Q
        - low_left.T
        - low_left
        + low_right) # (lower_Q - lower_Q.T).T @ (lower_Q - lower_Q.T)

    # MAT = np.eye(n+m+1) + Q.T @ Q
    u_star = np.linalg.solve(MAT, RHS)
    v_star = Q @ u_star
    tau, u1, u2 = u_star[n], u_star[:n], u_star[-m:]
    kappa, v2 = v_star[n], v_star[-m:] # we don't need v1
    ##########

    return tau, u1, u2, kappa, v2
