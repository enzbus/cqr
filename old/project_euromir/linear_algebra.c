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

#include "linear_algebra.h"

#define CSC_IMPL 3
#define CSR_IMPL 2 // for some reason it's faster (?)

#if CSC_IMPL == 1

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

    for (j = 0; j<n; j++)
        for (i = col_pointers[j]; i < col_pointers[j + 1]; i++)
            output[row_indexes[i]] += mult * (mat_elements[i] * input[j]);
};

#elif CSC_IMPL == 2

// profiling experiments: this one helps over the above
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

#elif CSC_IMPL == 3

void add_csc_matvec(
    const int n, /*number of columns*/
    const int * restrict col_pointers,
    const int * restrict row_indexes,
    const double * restrict mat_elements,
    double * restrict output,
    const double * restrict input,
    const double mult
    ){
    /*Could refactor, but easier to read with explicit row and col indexes.
    Should not matter, they're treated as counters in the for condition.
    
    In terms of mult, it's better to separate methods (add, sub, generic),
    since the sign/number they're called with is known at compile time. For
    now just keeping the one.
    */
    int j, i;

    double * restrict output_ptr;
    mat_elements += *col_pointers; // always 0?
    row_indexes += *col_pointers;

    for (j = 0; j<n; j++){
        for (i = *col_pointers; i < *(col_pointers + 1); i++){
            output_ptr = output + *row_indexes;
            (*output_ptr) += mult * ((*mat_elements) * (*input));
            mat_elements++;
            row_indexes++;
        }
        input++;
        col_pointers++;
    }
};
#endif
#if CSR_IMPL == 3

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

    const double * restrict input_ptr;
    mat_elements += *row_pointers; // always 0?
    col_indexes += *row_pointers;

    for (i = 0; i<m; i++){
        for (j = *row_pointers; j < *(row_pointers + 1); j++){
            input_ptr = input + *col_indexes;
            (*output) += mult * ((*mat_elements) * (*input_ptr));
            mat_elements++;
            col_indexes++;
        }
        output++;
        row_pointers++;
    }
};


#elif CSR_IMPL == 1

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

    for (i = 0; i<m; i++)
        for (j = row_pointers[i]; j < row_pointers[i + 1]; j++)
            output[i] += mult * (mat_elements[j] * input[col_indexes[j]]);

};

#elif CSR_IMPL == 2

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

#endif