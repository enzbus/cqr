import numpy as np
import scipy.sparse as sp

import project_euromir as lib

from .profiler import Profiler


class CSC(Profiler):

    def __init__(
            self, m=1000, n=1000, tries=100,
            complexities=np.linspace(.01, .25, 200)):
        self.m = m
        self.n = n

        self.input_vector = np.zeros(n)
        self.output_vector = np.zeros(m)
        self.mult = None
        self.matrix = None

        super().__init__(
            complexities=complexities, tries=tries,
            experiment_name=f'{self.__class__.__name__} matvec'
            + f' of a {self.m}x{self.n} matrix')

    def setup_at_complexity(self, complexity):
        """Method called to set up the experiment at given complexity.

        This is called once for each complexity value.
        """
        self.matrix = sp.random(
            m=self.m, n=self.n, dtype=float, density=complexity).tocsc()

    def prepare_sample_experiment(self, seed):
        """Method called repeatedly to set up each sample experiment.

        ``seed`` is ranged over ``tries``.
        """
        np.random.seed(seed)
        self.input_vector[:] = np.random.randn(self.n)
        # self.output_vector[:] = np.random.randn(self.m)
        # self.matrix.data[:] = np.random.randn(self.matrix.nnz)
        self.mult = np.random.choice([-1, 1, np.nan])
        if np.isnan(self.mult):
            self.mult = np.random.randn()

    def one_sample_experiment(self):
        """Method called repeatedly to run experiment.

        The call to this method is timed.
        """

        lib.add_csc_matvec(
            n=self.n, col_pointers=self.matrix.indptr,
            row_indexes=self.matrix.indices,
            mat_elements=self.matrix.data, input=self.input_vector,
            output=self.output_vector, mult=self.mult)

    @staticmethod
    def curve(x, slope, intercept):  # , square):
        """Function used for the curve fit. You can use any *args after x."""
        return slope*x + intercept  # + square * (x**2)

    curve_parameter_names = ('SLOPE', 'INTERCEPT')  # , 'SQUARE')


class CSR(CSC):
    """Variant to do CSR profiling."""

    def setup_at_complexity(self, complexity):
        """Method called to set up the experiment at given complexity.

        This is called once for each complexity value.
        """
        self.matrix = sp.random(
            m=self.m, n=self.n, dtype=float, density=complexity).tocsr()

    def one_sample_experiment(self):
        """Method called repeatedly to run experiment.

        The call to this method is timed.
        """

        lib.add_csr_matvec(
            m=self.m, row_pointers=self.matrix.indptr,
            col_indexes=self.matrix.indices,
            mat_elements=self.matrix.data, input=self.input_vector,
            output=self.output_vector, mult=self.mult)


if __name__ == '__main__':

    # print('CSC')
    # csc = CSC(
    #     m=1000, n=1000, tries=100, complexities=np.linspace(.01, .25, 200))
    # csc.run()

    print('CSR')
    csr = CSR(
        m=1000, n=1000, tries=100, complexities=np.linspace(.01, .25, 200))
    csr.run()
