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