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
"""Work out nonsymm-soc started some time ago (orig file in old_cqr).

After few attempts it appears better to use the one formulation which first
does bracket search and then Newton/bisection search.

Seems it works with extreme inputs values which is what I'm interested in.
"""

# if a has very large values (or 1./a in dual case) the algo below fails to give
# good result;

# rework the below, use instead the method with the inverse multiplier and the
# search variable in (1., np.inf)

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

N = 100

np.random.seed(0)
a = np.random.uniform(0, 1, size=N) #**5

# a[:3] = 1e-5 # this makes one of the two difficult
# a[:3] = 1e5 # this makes the other one difficult

# and this makes both difficult
a[:3] = 1e-5
a[-3:] = 1e5
print(np.sort(a))

y = np.random.randn(N)

def project_dual_case(t, y, a):
    """Project (t, y) on cone {(s,z) | s >= ||z * a||_2} in the case t < 0."""
    assert t < 0

    # mu is the Lagrangia multiplier of the inequality, s and z depend on it
    # mu is in (-inf, -0.5), s is always positive
    s = lambda mu: t / (1 + 2 * mu)
    z = lambda mu: y / (1 - 2 * mu * a**2)

    # this is the error which we want to be zero
    error = lambda mu: s(mu) - np.linalg.norm(z(mu) * a)

    # once replaced with Newton search we may not even need the bracketing

    # find brackets for the search
    high = -.5
    low = -1
    for _ in range(100):
        if error(low) > 0: # it appears the error is positive if mu too small
            high = low
            low *= 2.
        else:
            print(f"BRACKET SEARCH REQUIRED {_} ITERS")
            break
    else:
        raise ValueError("Bracket search failed!")

    # replace this with a Newton search
    result = sp.optimize.root_scalar(
        error, x0=(high + low)/2., bracket=(low, high),
        # rtol is around the minimum allowed by impl
        xtol=np.finfo(float).eps, rtol=4*np.finfo(float).eps)
    print(result)
    print("error =", error(result.root))
    projection = np.concatenate([[s(result.root)], z(result.root)])

    # assert projection on cone surface
    assert np.isclose(
        np.linalg.norm(projection[1:]*a),
        projection[0])

    # assert diff on dual cone surface
    orig_vector = np.concatenate([[t], y])
    assert np.isclose(
        np.linalg.norm((projection[1:]-orig_vector[1:])/a),
        projection[0]-orig_vector[0])

    return projection

###
# CASE 2: t negative but greater than -np.linalg.norm(y/a)
###

print('CASE T NEGATIVE AND WE PROJECT ON UPPER CONE')
t = -np.linalg.norm(y/a) / 2.
projection = project_dual_case(t=t, y=y, a=a)


###
# CASE 1: t positive and smaller than np.linalg.norm(y * a)
###
print('CASE T POSITIVE AND WE REFORMULATE WITH DUAL TO REUSE OTHER CLAUSE')

t = np.linalg.norm(y * a) / 2.

# we reformulate with dual
projection_dualized = project_dual_case(t=-t, y=-y, a=1./a)
input_vector = np.concatenate([[t], y])
projection = projection_dualized + np.concatenate([[t], y])

# double check to be safe
assert np.isclose(projection[0], np.linalg.norm(projection[1:] * a))
assert np.isclose(projection[0]-input_vector[0], np.linalg.norm((projection[1:]-input_vector[1:]) / a))



# breakpoint()

# # so we should instead use the approach with the inverse multiplier


# print("PRIMAL CASE")

# # t>0, t < np.linalg.norm(y * a)
# t = np.linalg.norm(y * a) / 2.

# # in this case mu is in (-.5, 0.)
# # s = lambda mu: t / (1 + 2 * mu)
# # z = lambda mu: y / (1 - 2 * mu * a**2)
# # error = lambda mu: s(mu)**2 - np.sum((z(mu)*a)**2)

# # in this case s(mu)>0
# # error = lambda mu: t / (1 + 2 * mu) - np.linalg.norm((y * a) / (1 - 2 * mu * a**2))
# # error = lambda mu: t - np.linalg.norm((1 + 2 * mu) * (y * a) / (1 - 2 * mu * a**2))

# # nu = -2 * mu
# # so nu in (0., 1.)
# # error = lambda nu: t - np.linalg.norm((1 - nu) * (y * a) / (1 + nu * a**2))
# error = lambda nu: t - np.linalg.norm((1 - nu) * y / (1./a + nu * a))


# result = sp.optimize.root_scalar(error, x0=.5, bracket=(0., 1.),
#     # rtol is the minimum allowed by impl
#     xtol=np.finfo(float).eps, rtol=4*np.finfo(float).eps)
# print(result)
# print("result in (0., 1.) =", result.root)
# print("error =", error(result.root))

# ###
# # case 2
# ###


# print("DUAL CASE")

# # # t<0, t > -np.linalg.norm(y/a)
# t = -np.linalg.norm(y/a) / 4.

# # in this case mu is in (-inf, -.5)
# s = lambda mu: t / (1 + 2 * mu)
# z = lambda mu: y / (1 - 2 * mu * a**2)
# # error = lambda mu: s(mu) - np.linalg.norm((z(mu)*a))
# error = lambda mu: t / (1 + 2 * mu) - np.linalg.norm(y / (1./a - 2 * mu * a))


# # find brackets
# high = -.5
# low = -1
# for _ in range(100):
#     if error(low) > 0:
#         high = low
#         low *= 2.
#     else:
#         break

# result = sp.optimize.root_scalar(error, x0=(high + low)/2., bracket=(low, high),
#     # rtol is the minimum allowed by impl
#     xtol=np.finfo(float).eps, rtol=4*np.finfo(float).eps)
# print(result)
# print("result in (0., 1.) =", result.root)
# print("error =", error(result.root))

# projection = np.concatenate([[s(result.root)], z(result.root)])

# # assert projection on cone surface
# assert np.isclose(np.linalg.norm(projection[1:]*a), projection[0])

# # assert diff on dual cone surface
# orig_vector = np.concatenate([[t], y])
# assert np.isclose(np.linalg.norm((projection[1:]-orig_vector[1:])/a), projection[0]-orig_vector[0])


# breakpoint()

# # (-t, -y) projected to dual cone
# newt = -t
# newy = -y
# newa = 1./a
# print(np.sort(newa))

# # check that after dualization we are indeed in case 1
# assert newt >0
# assert newt < np.linalg.norm(newy * newa)

# # so nu should be in  (0., 1.)
# # error = lambda nu: newt - np.linalg.norm((1 - nu) * (newy * newa) / (1 + nu * newa**2))
# error = lambda nu: newt - np.linalg.norm((1 - nu) * (newy) / (1./newa + nu * newa))


# result = sp.optimize.root_scalar(error, x0=.5, bracket=(0., 1.),
#     # rtol is the minimum allowed by impl
#     xtol=np.finfo(float).eps, rtol=4*np.finfo(float).eps)
# print(result)
# print("result in (0., 1.) =", result.root)
# print("error =", error(result.root))


# # breakpoint()

# # a_inv = 1./a
# # error_dual = lambda nu: -t - np.linalg.norm((1 - nu) * (-y * a_inv) / (1 + nu * a_inv**2))

# # result = sp.optimize.root_scalar(error_dual, x0=.5, bracket=(0., 1.),
# #     # rtol is the minimum allowed by impl
# #     xtol=np.finfo(float).eps, rtol=4*np.finfo(float).eps)
# # print(result)
# # print("result in (0., 1.) =", result.root)
# # print("error =", error_dual(result.root))


# # # in this case mu is in (-inf, -.5)
# # s = lambda mu: t / (1 + 2 * mu)
# # z = lambda mu: y / (1 - 2 * mu * a**2)
# # loss = lambda mu: s(mu)**2 - np.sum((z(mu)*a)**2)

# # # loss = lambda mu: s(mu) + np.linalg.norm(z(mu)*a)
# # print(loss(-1e6), loss(-1e10), loss(-1e12))
# # raise Exception
# # # j = 1. / (1 + 2 * mu)
# # # so j in (-.5, 0.)


# # # loss = lambda mu: t / (1 + 2 * mu) + np.linalg.norm((y * a) / (1 - 2 * mu * a**2))
# # # loss = lambda mu: t - np.linalg.norm((1 + 2 * mu) * (y * a) / (1 - 2 * mu * a**2))

# # # nu = 2 * mu
# # # so nu in (-inf, -1)
