/*
BSD 3-Clause License

Copyright (c) 2024-, Enzo Busseti

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "linear_algebra.h"

void add_csc_matvec(
    const int n, /*number of columns*/
    const int * col_pointers,
    const int * row_indexes,
    const double * mat_elements,
    double * output,
    const double * input,
    const double mult
    ){
    int j, i;
    if (mult == 1.0){ 
        for (j = 0; j<n; j++)
            for (i = col_pointers[j]; i < col_pointers[j + 1]; i++)
                output[row_indexes[i]] += mat_elements[i] * input[j];
    } else if (mult == -1.0){
        for (j = 0; j<n; j++)
            for (i = col_pointers[j]; i < col_pointers[j + 1]; i++)
                output[row_indexes[i]] -= mat_elements[i] * input[j];
    } else {
        for (j = 0; j<n; j++)
            for (i = col_pointers[j]; i < col_pointers[j + 1]; i++)
                output[row_indexes[i]] += mult * (mat_elements[i] * input[j]);
    }
};

void add_csr_matvec(
    const int m, /*number of rows*/
    const int * row_pointers, 
    const int * col_indexes,
    const double * mat_elements,
    double * output,
    const double * input,
    const double mult
    ){
    int j, i;
    if (mult == 1.0){ 
        for (i = 0; i<m; i++)
            for (j = row_pointers[i]; j < row_pointers[i + 1]; j++)
                output[i] +=  mat_elements[j] * input[col_indexes[j]];
    } else if (mult == -1.0){
        for (i = 0; i<m; i++)
            for (j = row_pointers[i]; j < row_pointers[i + 1]; j++)
                output[i] -=  mat_elements[j] * input[col_indexes[j]];
    } else {
        for (i = 0; i<m; i++)
            for (j = row_pointers[i]; j < row_pointers[i + 1]; j++)
                output[i] += mult * (mat_elements[j] * input[col_indexes[j]]);
    }
};