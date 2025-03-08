"""Quasi-Newton update for inverse Hessian."""

import numpy as np

def _psd_project(theta):
    eival, eivec = np.linalg.eigh(theta)
    return eivec @ np.diag(np.maximum(eival, -1.)) @ eivec.T

def quasi_newton(
    Y: np.array,
    S: np.array,
    grad: np.array,
    gamma: float,
    num_dr_iters=10,
    ):
    """Compute approximate inverse Hessian and apply it to gradient.
    
    :param Y: Each column is a recent gradient change.
    :type Y: np.array
    :param S: Each column is a recent step.
    :type S: np.array
    :param grad: Current gradient.
    :type grad: np.array
    :param gamma: Local curvature approximation, initial Hessian inverse is
        ``gamma`` times the identity.
    :type gamma: float
    :param num_dr_iters: Number of Douglas-Rachford iterates of PSD program.
    :type num_dr_iters: np.array
    """
    
    N, mem = Y.shape

    stacked_mat = np.hstack([Y, S])
    u, s, v = np.linalg.svd(stacked_mat, full_matrices=False)
    A = np.eye(2*mem) + np.diag(s) @ v[:, :mem] @ v[:, :mem].T @ np.diag(s)
    B = np.diag(s) @ v[:, :mem] @ (gamma * v[:, :mem] - v[:, mem:]).T @ np.diag(s)
    C = np.diag(s) @ v[:, :mem] @ v[:, :mem].T @ np.diag(s)
    new_s, V = np.linalg.eigh(C)
    SCALING_MATRIX = (np.ones((mem,mem)) * new_s).T + np.ones((mem,mem)) * new_s + 2 * np.ones((mem,mem))




if __name__ == '__main__':
    # temporary test

    # sizes
    N = 20
    mem = 2

    # these are the matrices representing steps and gradient changes
    np.random.seed(0)
    Y = np.random.randn(N, mem)*1e-5
    S = np.random.randn(N, mem)*1e-5

    # define H0
    gamma = 1 # diagonal of H0
    H0 = np.eye(N) * gamma

    # regularizer
    # alpha = 1 # scaler of first obj term / regularizer