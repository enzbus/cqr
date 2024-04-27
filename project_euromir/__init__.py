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
import pathlib
import platform
import numpy as _np
import ctypes

_EXTS = {'Linux': '.so', 'Darwin': '.dylib', 'Windows': '.dll'}
_T = {
    'int': ctypes.c_int, 
    'int*': ctypes.POINTER(ctypes.c_int),
    'double': ctypes.c_double,
    'double*': ctypes.POINTER(ctypes.c_double),
    'bool': ctypes.c_bool, 
    }

for fname in pathlib.Path(__file__).parent.iterdir():
    if fname.suffix == _EXTS[platform.system()]:
        print('LOADING LIBRARY', fname)
        LIBRARY = ctypes.cdll.LoadLibrary(fname)

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
        col_pointers.ctypes.data_as(_T['int*']),
        row_indexes.ctypes.data_as(_T['int*']),
        mat_elements.ctypes.data_as(_T['double*']),
        output.ctypes.data_as(_T['double*']),
        input.ctypes.data_as(_T['double*']),
        _T['bool'](sign_plus))
