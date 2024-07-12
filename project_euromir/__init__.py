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
"""project_euromir."""

__version__ = '0.0.1'

import ctypes as _ctypes
import pathlib as _pathlib
import platform as _platform

import numpy as _np

##
# Load library
##

_EXTS = {
    'Linux': '.so',
    'Darwin': '.dylib',
    'Windows': '.dll',
}

for _fname in _pathlib.Path(__file__).parent.iterdir():
    if _fname.suffix == _EXTS[_platform.system()]:
        print('LOADING LIBRARY', _fname)
        LIBRARY = _ctypes.cdll.LoadLibrary(str(_fname))
        break
else:
    raise ImportError(
        'Could not load the compiled library!')

##
# Utilities for interfacing via ctypes
##

_T = {  # ctypes
    'int': _ctypes.c_int,
    'int*': _ctypes.POINTER(_ctypes.c_int),
    'double': _ctypes.c_double,
    'double*': _ctypes.POINTER(_ctypes.c_double),
    'bool': _ctypes.c_bool,
}

_NT = {  # np.dtypes
    'int': _np.int32,
    'double': _np.float64,
    'bool': bool
}


def _ndarray_to_pointer(ndarray, c_type):
    """1-dimensional Numpy array to pointer."""
    assert isinstance(ndarray, _np.ndarray)
    assert len(ndarray.ctypes.shape) == 1
    assert len(ndarray.ctypes.strides) == 1
    oldptr = ndarray.ctypes.data
    result = ndarray.astype(  # only copies if conversion required
        dtype=_NT[c_type], order='C', copy=False)
    assert result.ctypes.data == oldptr  # relax if required
    return result.ctypes.data_as(_T[c_type + '*'])  # no copy


def _python_to_c(obj, c_type):
    """Convert Python scalars or simple Numpy arrays to C objects."""
    if c_type[-1] == '*':
        return _ndarray_to_pointer(obj, c_type[:-1])
    return _T[c_type](obj)


def _interface_function(function_name, args, returns=None):
    """Interface via (Numpy) ctypes a void function."""
    assert hasattr(LIBRARY, function_name)

    getattr(LIBRARY, function_name).argtypes = [_T[el[1]] for el in args]
    getattr(LIBRARY, function_name).restype = None

    def fun(**kwargs):
        funargs = [_python_to_c(kwargs[arg], c_type) for arg, c_type in args]
        if returns is not None:
            getattr(LIBRARY, function_name).restype = _T[returns]
        return getattr(LIBRARY, function_name)(*funargs)

    # add documentation
    fun.__doc__ = f"{function_name}\n\n"
    for arg, c_type in args:
        fun.__doc__ += f':param {arg}:\n'
        fun.__doc__ += f':type {arg}: {c_type}\n'
    if returns is not None:
        fun.__doc__ += f':rtype: {returns}\n'

    return fun


##
# Interface to functions
##

add_csc_matvec = _interface_function(
    function_name='add_csc_matvec',
    args=(
        ('n', 'int'),
        ('col_pointers', 'int*'),
        ('row_indexes', 'int*'),
        ('mat_elements', 'double*'),
        ('output', 'double*'),
        ('input', 'double*'),
        ('mult', 'double'),
    )
)

add_csr_matvec = _interface_function(
    function_name='add_csr_matvec',
    args=(
        ('m', 'int'),
        ('row_pointers', 'int*'),
        ('col_indexes', 'int*'),
        ('mat_elements', 'double*'),
        ('output', 'double*'),
        ('input', 'double*'),
        ('mult', 'double'),
    )
)

dcsrch = _interface_function(
    function_name='dcsrch',
    args=(
        ('stp', 'double*'),
        ('f', 'double*'),
        ('g', 'double*'),
        ('ftol', 'double*'),
        ('gtol', 'double*'),
        ('xtol', 'double*'),
        ('stpmin', 'double*'),
        ('stpmax', 'double*'),
        ('isave', 'int*'),
        ('dsave', 'double*'),
        ('start', 'bool'),
    ),
    returns=('int')
)

###
# Main function
###

from .solver import solve

###
# CVXPY interface
###

try:
    from .cvxpy_solver import Solver
except ImportError:
    pass
