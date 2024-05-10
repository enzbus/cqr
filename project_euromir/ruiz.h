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

/*
Algorithm (Ruiz scaling) is described here:
https://web.stanford.edu/~takapoui/preconditioning.pdf

It is applied to the matrix:

[ A   b]
[ c.T 0]

Naming convention is described in section 5 here:
https://web.stanford.edu/~boyd/papers/pdf/scs.pdf

e, d, rho, and sigma are taken as starting values,
the caller should initialize them to 1. to prevent warm starting
*/

void ruiz_l2_equilibrate(
    int m, /*number of rows*/
    int n, /*number of columns*/

    const double * restrict b, /* len m */
    const double * restrict c, /* len n */ 
    const int * restrict col_pointers, /*CSC matrix*/
    const int * restrict row_indexes, /*CSC matrix*/
    const double * restrict mat_elements, /* len nnz */

    double * restrict b_transformed, /* len m */
    double * restrict c_transformed, /* len n */
    double * restrict mat_elements_transformed, /* len nnz */
    double * restrict d, /*len m, row scaler*/
    double * restrict e, /*len n, columns scaler*/
    double * restrict sigma, /*len 1*/
    double * restrict rho, /*len 1*/

    const double eps_1,
    const double eps_2,
    const int max_iters,
);