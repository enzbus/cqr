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
#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <stdbool.h>
 
/*
output += mult * (csc_matrix @ input)
*/
void add_csc_matvec(
    const int n, /*number of columns*/
    const int * restrict col_pointers,
    const int * restrict row_indexes,
    const double * restrict mat_elements,
    double * restrict output,
    const double * restrict input,
    const double mult
    );

/*
output += mult * (csr_matrix @ input)
*/
void add_csr_matvec(
    const int m, /*number of rows*/
    const int * restrict row_pointers, 
    const int * restrict col_indexes,
    const double * restrict mat_elements,
    double * restrict output,
    const double * restrict input,
    const double mult
    );

#endif