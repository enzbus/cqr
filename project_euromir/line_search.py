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
"""Line search for convex loss using cubic splines."""

import numpy as np


class LineSearchError(Exception):
    """Error with the line search procedure."""

def line_search(function, gradient, x0=0., step0=1.):
    """Line search for convex loss using cubic splines."""
    print(f'Called line search with interval [{x0:.5e}, {step0:.5e}]')
    x_left = x0
    x_right = step0
    loss_left = function(x_left)
    loss_right = function(x_right)
    grad_left = gradient(x_left)
    if grad_left >= 0:
        raise LineSearchError('Left gradient is not negative!')
    grad_right = gradient(x_right)
    try:
        a, b, c, d = fit_cubspline(
            x_left=x_left, loss_left=loss_left, grad_left=grad_left,
            x_right=x_right, loss_right=loss_right, grad_right=grad_right)
    except np.linalg.LinAlgError as e:
        raise LineSearchError('Singular matrix!') from e

    root_right, root_left = get_roots_spline(a, b, c, d)

    print(f'Found roots ({root_left:.5e}, {root_right:.5e})')

    if grad_right > 0: # we must stay in the interval
        left_good = (root_left >= x_left) and (root_left <= x_right)
        right_good = (root_right >= x_left) and (root_right <= x_right)
        if left_good and right_good:
            raise LineSearchError('Found two roots in interval!')
        if not (left_good or right_good):
            raise LineSearchError('Found no roots in interval!')
        if left_good:
            next_point = root_left
        else:
            next_point = root_right

    if grad_right < 0: # we must go more right
        left_good = (root_left > x_right)
        right_good = (root_right > x_right)
        # both roots is ok, we pick the left one
        if not (left_good or right_good):
            raise LineSearchError('Found no candidate roots!')
        if left_good:
            next_point = root_left
        else:
            next_point = root_right

    print('next point', next_point)

    if next_point in [x0, step0]:
        return next_point

    g = gradient(next_point)
    print(f'abs gradient of next point: {np.abs(g):.2e}')

    if np.abs(g) < np.finfo(float).eps:
        print('Converged!!!')
        return next_point

    else:
        try:
            if g < 0:
                print('setting next point as left end')
                return line_search(function, gradient, x0=next_point, step0=step0)
            else:
                print('setting next point as right end')
                return line_search(function, gradient, x0=x0, step0=next_point)
        except (LineSearchError, AssertionError):
            print('Line search failed, return best point found...')
            print('loss', function(next_point))
            print('gradient', gradient(next_point))
            return next_point


def fit_cubspline(
        x_left, loss_left, grad_left, x_right, loss_right, grad_right):
    """Fit cubic spline."""
    # a * x^3 + b * x^2 + c * x + d
    assert grad_left < 0
    # assert grad_right > 0
    assert x_left < x_right
    matrix = [
        [x_left**3, x_left**2, x_left, 1.],
        [x_right**3, x_right**2, x_right, 1.],
        [3 * x_left**2, 2 * x_left, 1., 0.],
        [3 * x_right**2, 2 * x_right, 1., 0.],
    ]
    rhs = [loss_left, loss_right, grad_left, grad_right]
    return np.linalg.solve(matrix, rhs)

def get_roots_spline(a, b, c, d):
    # 3 * a * x^2 + 2 * b * x + c == 0
    tmp = (2*b)**2 - 4 * (3 * a * c)
    if tmp < 0:
        raise LineSearchError('Negative value in square root!')
    tmp = np.sqrt(tmp)
    # x = (-b ± √ (b2 - 4ac) )/2a
    if a == 0.:
        raise LineSearchError('Zero quadratic coefficient!')
    return (- 2 * b + tmp)/(6*a), (- 2 * b - tmp)/(6*a)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    def _function(x):
        return (x - 3)**2 + 5*(max(x - 1, 0))**2

    def _gradient(x):
        return 2 * (x-3) + 10 * max(x-1, 0)

    # xs = np.linspace(0,10)
    # plt.plot(xs, [_function(x) for x in xs])
    # plt.yscale('log')
    # plt.show()
    LEFT = 0.
    RIGHT = 1.
    # a, b, c, d = fit_cubspline(
    #     x_left=LEFT, loss_left=_function(LEFT), grad_left=_gradient(LEFT),
    #     x_right=RIGHT, loss_right=_function(RIGHT),
    #     grad_right=_gradient(RIGHT))

    # roots = get_roots_spline(a,b,c,d)
    # print(roots)

    next_point = line_search(_function, _gradient, x0=0., step0=1.)
