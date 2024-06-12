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

Header file for select functions translated from Lbfgsb3.0, see copyright
notice therein; it's licensed BSD 3-Clause.

Translation has been done with f2c and edited by hand. Including here also some
macros defined in f2c.h.
*/

#ifndef FORTRAN_H
#define FORTRAN_H

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))

void dcstep(
	double *stx,
	double *fx,
	double *dx, 
	double *sty,
	double *fy,
	double *dy,
	double *stp, 
	double *fp,
	double *dp,
	bool *brackt,
	double *stpmin, 
	double *stpmax);
    /*Function called inside the line search procedure, see its file for
    documentation from the original FORTRAN. No variable has been changed.*/

int dcsrch(
	double *stp,
	double *f,
	double *g, 
	double *ftol,
	double *gtol,
	double *xtol,
	double *stpmin,
	double *stpmax,
	int *isave,
	double *dsave,
	bool start
	);
	/*Line search function. We modified the signature; here are the differences
	with the original FORTRAN. The boolean start must be set to true on initial
	invocation and false afterwards. The return code is negative on input
	errors (each negative number corresponds to one error message from the
	original), 0 on successuful convergence, 1 on request to call again with
	a new evaluation ("FG"), and greater than 1 for warning messages that mean
	that the search failed and the current point is the best.
	*/

#endif