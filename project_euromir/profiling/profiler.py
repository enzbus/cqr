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
"""Profiler utility class."""


import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class Profiler:
    """Profiler utility class."""

    def __init__(self, complexities, tries, experiment_name):
        self._complexities = complexities
        self._tries = tries
        self._timers = np.empty(tries, dtype=float)
        self._meantimes = np.empty(len(complexities), dtype=float)
        self._stdtimes = np.empty(len(complexities), dtype=float)
        self._curve_args = None
        self._curve_args_cov = None
        self._experiment_name = experiment_name

    def setup_at_complexity(self, complexity):
        """Method called to set up the experiment at given complexity.

        This is called once for each complexity value. Seeding is done outside.
        """
        raise NotImplementedError

    def prepare_sample_experiment(self, seed):
        """Method called repeatedly to set up each sample experiment.

        ``seed`` is ranged over ``tries``.
        """
        raise NotImplementedError

    def one_sample_experiment(self):
        """Method called repeatedly to run experiment.

        The call to this method is timed.
        """
        raise NotImplementedError

    def run(self):
        """Run suite of experiments.

        This is 2 nested for loops.
        """
        for idx, complexity in enumerate(self._complexities):

            np.random.seed(idx)
            self.setup_at_complexity(complexity=complexity)

            for seed in range(self._tries):

                self.prepare_sample_experiment(seed=seed)

                t = time.time()
                self.one_sample_experiment()
                self._timers[seed] = time.time() - t

            # remove outliers (garbage collector)
            self._timers[:] = np.sort(self._timers)
            self._meantimes[idx] = np.mean(
                self._timers[self._tries//10:-(self._tries//10)])
            self._stdtimes[idx] = np.std(
                self._timers[self._tries//10:-(self._tries//10)])

        self.fit_curve()
        self.print_result()
        self.plot_result()

    @staticmethod
    def curve(x, *args):
        """Function used for the curve fit.

        You can use any *args after x.
        """
        raise NotImplementedError

    curve_parameter_names = None  # string tuple

    def fit_curve(self):
        """Fit curve to data at the end of the experiments."""
        # pylint: disable=unbalanced-tuple-unpacking
        self._curve_args, self._curve_args_cov = curve_fit(
            self.curve, xdata=self._complexities, ydata=self._meantimes,
            sigma=self._stdtimes,
            absolute_sigma=True  # this should do proper error propagation
        )

    def _sample_noisy_curve_args(self, seed):
        """Sample a noisy choice of curve args."""
        np.random.seed(seed)
        return np.random.multivariate_normal(
            mean=self._curve_args, cov=self._curve_args_cov)

    def print_result(self):
        """Print result of the curve fit."""

        for par, parstd, name in zip(
                self._curve_args,
                np.sqrt(np.diag(self._curve_args_cov)),
                self.curve_parameter_names):

            print(
                f'Curve parameter {name} is fitted in std interval '
                + f'[{par-parstd:.2e}, {par+parstd:.2e}]; mean is {par:.2e}')

    def plot_result(self):
        """Plot result of the curve fit."""

        plt.figure(figsize=(20, 12))
        plt.plot(self._complexities, self.curve(
            self._complexities, *self._curve_args), color='k',
            label='Best curve fit')
        _low, _high = plt.ylim()
        plt.errorbar(
            x=self._complexities, y=self._meantimes, yerr=self._stdtimes,
            fmt='o', color='r', label='Experiments at complexity')
        plt.ylim(_low, _high)
        plt.title(self._experiment_name)

        # fancy stuff; let's sample a few curves from the fitted mu, Sigma
        for _ in range(100):
            _args = self._sample_noisy_curve_args(_)
            plt.plot(self._complexities, self.curve(
                self._complexities, *_args), color='k', linestyle='--',
                alpha=.05, label='Noisy curve fits' if _ == 0 else None)

        plt.legend()
        plt.ylabel('mean plusminus std time per experiment, seconds')
        plt.xlabel('complexity parameter')
        plt.show()
