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

import ctypes
import pathlib
import platform

import numpy as _np

##
# Load library
##

_EXTS = {'Linux': '.so', 'Darwin': '.dylib', 'Windows': '.dll'}

for fname in pathlib.Path(__file__).parent.iterdir():
    if fname.suffix == _EXTS[platform.system()]:
        print('LOADING LIBRARY', fname)
        LIBRARY = ctypes.cdll.LoadLibrary(fname)
        break
else:
    raise ImportError(
        'Could not load the compiled library!')

##
# Utilities for interfacing via ctypes
##

_T = {  # ctypes
    'int': ctypes.c_int,
    'int*': ctypes.POINTER(ctypes.c_int),
    'double': ctypes.c_double,
    'double*': ctypes.POINTER(ctypes.c_double),
    'bool': ctypes.c_bool,
}

_NT = {  # np.dtypes
    'int': _np.int32,
    'double': _np.float64,
    'bool': bool
}


def _ndarray_to_pointer(ndarray, c_type):
    """1-dimensional Numpy array to pointer."""
    assert len(ndarray.ctypes.shape) == 1
    assert len(ndarray.ctypes.strides) == 1
    oldptr = ndarray.ctypes.data
    result = ndarray.astype(  # only copies if conversion required
        dtype=_NT[c_type], order='C', copy=False)
    assert result.ctypes.data == oldptr  # relax if required
    return result.ctypes.data_as(_T[c_type + '*'])  # no copy


##
# Interface to functions
##

assert hasattr(LIBRARY, 'csc_matvec')


LIBRARY.csc_matvec.argtypes = [
    _T['int'],
    _T['int*'],
    _T['int*'],
    _T['double*'],
    _T['double*'],
    _T['double*'],
    _T['bool'],
]
LIBRARY.csc_matvec.restype = None


def csc_matvec(
        n, col_pointers, row_indexes, mat_elements, output, input, sign_plus):
    """csc matvec"""

    LIBRARY.csc_matvec(
        _T['int'](n),
        _ndarray_to_pointer(col_pointers, 'int'),
        _ndarray_to_pointer(row_indexes, 'int'),
        _ndarray_to_pointer(mat_elements, 'double'),
        _ndarray_to_pointer(output, 'double'),
        _ndarray_to_pointer(input, 'double'),
        _T['bool'](sign_plus))
