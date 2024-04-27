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
"""project_euromir"""

__version__ = '0.0.1'

import ctypes as _ctypes
import pathlib as _pathlib
import platform as _platform

import numpy as _np

##
# Load library
##

_EXTS = {'Linux': '.so', 'Darwin': '.dylib', 'Windows': '.dll'}

for _fname in _pathlib.Path(__file__).parent.iterdir():
    if _fname.suffix == _EXTS[_platform.system()]:
        print('LOADING LIBRARY', _fname)
        LIBRARY = _ctypes.cdll.LoadLibrary(_fname)
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


def _interface_function(function_name, args):
    """Interface via (Numpy) ctypes a void function."""
    assert hasattr(LIBRARY, function_name)

    getattr(LIBRARY, function_name).argtypes = [_T[el[1]] for el in args]
    getattr(LIBRARY, function_name).restype = None

    def fun(**kwargs):
        funargs = [_python_to_c(kwargs[arg], c_type) for arg, c_type in args]
        return getattr(LIBRARY, function_name)(*funargs)

    fun.__doc__ = f"{function_name}\n\n"
    for arg, c_type in args:
        fun.__doc__ += f':param {arg}:\n'
        fun.__doc__ += f':type {arg}: {c_type}\n'

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
