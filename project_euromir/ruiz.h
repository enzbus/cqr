/*
Copyright 2024 Enzo Busseti

This file is part of Project Euromir.

Project Euromir is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

Project Euromir is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
Project Euromir. If not, see <https://www.gnu.org/licenses/>.
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