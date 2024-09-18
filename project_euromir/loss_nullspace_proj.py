"""Loss and related functions for nullspace projection model."""

import numpy as np


class NullSpaceModel:

    def __init__(
        self, m, n, zero, nonneg, matrix_transfqr, b, c, nullspace_projector):
        self.m = m
        self.n = n
        self.zero = zero
        self.nonneg = nonneg
        self.matrix = matrix_transfqr,
        self.b = b
        self.c = c
        self.nullspace_projector = nullspace_projector

        # since matrix is from QR
        self.y0 = -self.c @ self.matrix.T

    def loss(self, variable):
        x = variable[:self.n]; y_null = variable[self.n:]
        y = self.y0 + self.nullspace_projector @ y_null
        s = -self.matrix @ x + self.b
        y_loss = np.linalg.norm(np.minimum(y[self.zero:], 0.))**2
        s_loss_zero = np.linalg.norm(s[:self.zero])**2
        s_loss_nonneg = np.linalg.norm(np.minimum(s[self.zero:], 0.))**2
        gap_loss = (self.c.T @ x + self.b.T @ y)**2
        return (y_loss + s_loss_zero + s_loss_nonneg + gap_loss) / 2.

    def gradient(self, variable):
        x = variable[:self.n]; y_null = variable[self.n:]
        y = self.y0 + self.nullspace_projector @ y_null
        s = -self.matrix @ x + self.b
