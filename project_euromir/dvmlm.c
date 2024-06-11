/* dvmlm.f -- translated by f2c (version 20200916).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"

/* Table of constant values */

static integer c__1 = 1;
static doublereal c_b35 = -1.;

/* Subroutine */ int dvmlm_(integer *n, doublereal *x, doublereal *f, 
	doublereal *fgrad, doublereal *frtol, doublereal *fatol, doublereal *
	fmin, char *task, integer *m, doublereal *s, doublereal *y, 
	doublereal *rho, integer *isave, doublereal *dsave, doublereal *wa1, 
	doublereal *wa2, ftnlen task_len)
{
    /* System generated locals */
    integer s_dim1, s_offset, y_dim1, y_offset, i__1, i__2, i__3;
    doublereal d__1;

    /* Builtin functions */
    integer s_cmp(char *, char *, ftnlen, ftnlen);
    /* Subroutine */ int s_copy(char *, char *, ftnlen, ftnlen);

    /* Local variables */
    static integer i__;
    static doublereal f0, gd, gd0, stp;
    static integer mark;
    extern doublereal ddot_(integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    static integer iter;
    static char work[30];
    extern doublereal dnrm2_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    static doublereal scale;
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *, 
	    doublereal *, integer *), dlmmv_(integer *, integer *, doublereal 
	    *, doublereal *, doublereal *, doublereal *, integer *, 
	    doublereal *, doublereal *), daxpy_(integer *, doublereal *, 
	    doublereal *, integer *, doublereal *, integer *);
    static doublereal sftol, sgtol, sxtol;
    extern /* Subroutine */ int dcsrch_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, char *, 
	    doublereal *, doublereal *, integer *, doublereal *, ftnlen);
    static doublereal stpmin, stpmax;

/*     ********** */

/*     Subroutine dvmlm */

/*     This subroutine computes a local minimizer of a function */
/*     of n variables by a limited memory variable metric method. */
/*     The user must evaluate the function and the gradient. */

/*     This subroutine uses reverse communication. */
/*     The user must choose an initial approximation x to the */
/*     minimizer, evaluate the function and the gradient at x, */
/*     and make the initial call with task set to 'START'. */
/*     On exit task indicates the required action. */

/*     A typical invocation of dvmlm has the following outline: */

/*     Choose a starting vector x. */
/*     Evaluate the function at x; store in f. */
/*     Evaluate the gradient at x; store in fgrad. */

/*     task = 'START' */
/*  10 continue */
/*        call dvmlm(n,x,f,fgrad,frtol,fatol,fmin,task,m,s,y,rho, */
/*                   isave,dsave,wa1,wa2) */
/*        if (task .eq. 'FG') then */
/*           Evaluate the function at x; store in f. */
/*           Evaluate the gradient at x; store in fgrad. */
/*           go to 10 */
/*        else if (task .eq. 'NEWX') then */
/*           The approximation x, function f, and gradient fgrad */
/*           are available for inspection. */
/*           go to 10 */
/*        end if */

/*     NOTE: The user must not alter work arrays between calls. */

/*     The subroutine statement is */

/*       subroutine dvmlm(n,x,f,fgrad,frtol,fatol,fmin,task,m,s,y,rho, */
/*                        isave,dsave,wa1,wa2) */

/*     where */

/*       n is an integer variable. */
/*         On entry n is the number of variables. */
/*         On exit n is unchanged. */

/*       x is a double precision array of dimension n. */
/*         On entry x is an approximation to the solution. */
/*         On exit x is the current approximation. */

/*       f is a double precision variable. */
/*         On entry f is the value of the function at x. */
/*         On final exit f is the value of the function at x. */

/*       fgrad is a double precision array of dimension n. */
/*         On entry fgrad is the value of the gradient at x. */
/*         On final exit fgrad is the value of the gradient at x. */

/*       frtol is a double precision variable. */
/*         On entry frtol specifies the relative error desired in the */
/*            function. Convergence occurs if the estimate of the */
/*            relative error between f(x) and f(xsol), where xsol */
/*            is a local minimizer, is less than frtol. */
/*         On exit frtol is unchanged. */

/*       fatol is a double precision variable. */
/*         On entry fatol specifies the absolute error desired in the */
/*            function. Convergence occurs if the estimate of the */
/*            absolute error between f(x) and f(xsol), where xsol */
/*            is a local minimizer, is less than fatol. */
/*         On exit fatol is unchanged. */

/*       fmin is a double precision variable. */
/*         On entry fmin specifies a lower bound for the function. */
/*            The subroutine exits with a warning if f < fmin. */
/*         On exit fmin is unchanged. */

/*       task is a character variable of length at least 60. */
/*         On initial entry task must be set to 'START'. */
/*         On exit task indicates the required action: */

/*            If task(1:2) = 'FG' then evaluate the function and */
/*            gradient at x and call dvmlm again. */

/*            If task(1:4) = 'NEWX' then a new iterate has been */
/*            computed. The approximation x, function f, and */
/*            gradient fgrad are available for examination. */

/*            If task(1:4) = 'CONV' then the search is successful. */

/*            If task(1:4) = 'WARN' then the subroutine is not able */
/*            to satisfy the convergence conditions. The exit value */
/*            of x contains the best approximation found. */

/*            If task(1:5) = 'ERROR' then there is an error in the */
/*            input arguments. */

/*         On exit with convergence, a warning or an error, the */
/*            variable task contains additional information. */

/*       m is an integer variable. */
/*          On entry m specifies the amount of storage. */
/*          On exit m is unchanged. */

/*       s is a double precision work array of dimension (n,m). */

/*       y is a double precision work array of dimension (n,m). */

/*       rho is a double precision work array of dimension m. */

/*       isave is an integer work array of dimension 5. */

/*       dsave is a double precision work array of dimension 24. */

/*       wa1 is a double precision work array of dimension n. */

/*       wa2 is a double precision work array of dimension m. */

/*     Subprograms called */

/*       MINPACK-2 ... dcsrch, dlmmv */

/*       Level 1 BLAS ... daxpy, dcopy, ddot, dnrm2, dscal */

/*     MINPACK-2 Project. April 1995. */
/*     Argonne National Laboratory and University of Minnesota. */
/*     Brett M. Averick, Richard G. Carter, and Jorge J. More'. */

/*     ********** */
    /* Parameter adjustments */
    --wa1;
    --fgrad;
    --x;
    --wa2;
    --rho;
    y_dim1 = *n;
    y_offset = 1 + y_dim1;
    y -= y_offset;
    s_dim1 = *n;
    s_offset = 1 + s_dim1;
    s -= s_offset;
    --isave;
    --dsave;

    /* Function Body */
    if (s_cmp(task, "START", (ftnlen)5, (ftnlen)5) == 0) {
/*        Check the input arguments for errors. */
	if (*n <= 0) {
	    s_copy(task, "ERROR: N .LE. 0", task_len, (ftnlen)15);
	}
	if (*m <= 0) {
	    s_copy(task, "ERROR: M .LE. 0", task_len, (ftnlen)15);
	}
	if (*frtol <= 0.) {
	    s_copy(task, "ERROR: FRTOL .LE. 0", task_len, (ftnlen)19);
	}
	if (*fatol <= 0.) {
	    s_copy(task, "ERROR: FATOL .LE. 0", task_len, (ftnlen)19);
	}
	if (*f <= *fmin) {
	    s_copy(task, "ERROR: INITIAL F .LE. FMIN", task_len, (ftnlen)26);
	}
/*        Exit if there are errors on input. */
	if (s_cmp(task, "ERROR", (ftnlen)5, (ftnlen)5) == 0) {
	    return 0;
	}
/*        Initialize local variables. */
	iter = 1;
	mark = 1;
/*        Initialize step information. */
	scale = dnrm2_(n, &fgrad[1], &c__1);
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    s[i__ + s_dim1] = fgrad[i__] / scale;
/* L10: */
	}
/*        Initialize line search parameters. */
	sftol = .001;
	sgtol = .9;
	sxtol = .1;
/*        Set work to start the search. */
	s_copy(work, "START SEARCH", (ftnlen)30, (ftnlen)12);
    } else {
/*        Restore local variables. */
	if (isave[1] == 1) {
	    s_copy(work, "SEARCH", (ftnlen)30, (ftnlen)6);
	}
	if (isave[1] == 2) {
	    s_copy(work, "SEARCH DIRECTION", (ftnlen)30, (ftnlen)16);
	}
	iter = isave[2];
	mark = isave[3];
	sftol = dsave[1];
	sgtol = dsave[2];
	sxtol = dsave[3];
	f0 = dsave[4];
	gd = dsave[5];
	gd0 = dsave[6];
	stp = dsave[7];
	stpmin = dsave[8];
	stpmax = dsave[9];
	scale = dsave[10];
    }
L20:
    if (s_cmp(work, "START SEARCH", (ftnlen)30, (ftnlen)12) == 0) {
/*        Initialize the line search subroutine. */
	f0 = *f;
	stp = 1.;
	gd0 = -ddot_(n, &fgrad[1], &c__1, &s[mark * s_dim1 + 1], &c__1);
	stpmin = 0.;
	stpmax = (*fmin - f0) / (sgtol * gd0);
	stp = min(stp,stpmax);
	dcopy_(n, &x[1], &c__1, &wa1[1], &c__1);
	dcopy_(n, &fgrad[1], &c__1, &y[mark * y_dim1 + 1], &c__1);
	s_copy(task, "START SEARCH", task_len, (ftnlen)12);
	s_copy(work, "SEARCH", (ftnlen)30, (ftnlen)6);
    }
    if (s_cmp(work, "SEARCH", (ftnlen)30, (ftnlen)6) == 0) {
/*        Determine the line search parameter. */
	if (*f < *fmin) {
	    s_copy(task, "WARNING: F .LT. FMIN", task_len, (ftnlen)20);
	    goto L30;
	}
	gd = -ddot_(n, &fgrad[1], &c__1, &s[mark * s_dim1 + 1], &c__1);
	dcsrch_(&stp, f, &gd, &sftol, &sgtol, &sxtol, task, &stpmin, &stpmax, 
		&isave[4], &dsave[11], task_len);
/*        Compute the new iterate. */
	dcopy_(n, &wa1[1], &c__1, &x[1], &c__1);
	d__1 = -stp;
	daxpy_(n, &d__1, &s[mark * s_dim1 + 1], &c__1, &x[1], &c__1);
/*        Continue if the line search has converged. */
	if (s_cmp(task, "CONV", (ftnlen)4, (ftnlen)4) != 0 && s_cmp(task, 
		"WARNING: XTOL TEST SATISFIED", task_len, (ftnlen)28) != 0) {
	    goto L30;
	}
/*        Compute the step and gradient change. */
	++iter;
	daxpy_(n, &c_b35, &fgrad[1], &c__1, &y[mark * y_dim1 + 1], &c__1);
	dscal_(n, &stp, &s[mark * s_dim1 + 1], &c__1);
	rho[mark] = ddot_(n, &y[mark * y_dim1 + 1], &c__1, &s[mark * s_dim1 + 
		1], &c__1);
/*        Compute the scale. */
	if (rho[mark] > 0.) {
	    scale = rho[mark] / ddot_(n, &y[mark * y_dim1 + 1], &c__1, &y[
		    mark * y_dim1 + 1], &c__1);
	} else {
	    scale = 1.;
	}
/*        Set task to signal a new iterate. */
/*        Set work to compute a new search direction. */
	s_copy(task, "NEWX", task_len, (ftnlen)4);
	s_copy(work, "SEARCH DIRECTION", (ftnlen)30, (ftnlen)16);
/*        Test for convergence. */
	if ((d__1 = *f - f0, abs(d__1)) <= *fatol && stp * abs(gd0) <= *fatol)
		 {
	    s_copy(task, "CONVERGENCE: FATOL TEST SATISFIED", task_len, (
		    ftnlen)33);
	}
	if ((d__1 = *f - f0, abs(d__1)) <= *frtol * abs(f0) && stp * abs(gd0) 
		<= *frtol * abs(f0)) {
	    s_copy(task, "CONVERGENCE: FRTOL TEST SATISFIED", task_len, (
		    ftnlen)33);
	}
	goto L30;
    }
    if (s_cmp(work, "SEARCH DIRECTION", (ftnlen)30, (ftnlen)16) == 0) {
/*        Compute -H*g. */
	dcopy_(n, &fgrad[1], &c__1, &wa1[1], &c__1);
/* Computing MIN */
	i__2 = *m, i__3 = iter - 1;
	i__1 = min(i__2,i__3);
	dlmmv_(n, &i__1, &s[s_offset], &y[y_offset], &rho[1], &scale, &mark, &
		wa1[1], &wa2[1]);
	++mark;
	if (mark == *m + 1) {
	    mark = 1;
	}
	dcopy_(n, &wa1[1], &c__1, &s[mark * s_dim1 + 1], &c__1);
/*        Set task and work to initialize the line search. */
	s_copy(task, "START SEARCH", task_len, (ftnlen)12);
	s_copy(work, "START SEARCH", (ftnlen)30, (ftnlen)12);
	goto L20;
    }
L30:
/*     Save local variables. */
    if (s_cmp(work, "SEARCH", (ftnlen)30, (ftnlen)6) == 0) {
	isave[1] = 1;
    }
    if (s_cmp(work, "SEARCH DIRECTION", (ftnlen)30, (ftnlen)16) == 0) {
	isave[1] = 2;
    }
    isave[2] = iter;
    isave[3] = mark;
    dsave[1] = sftol;
    dsave[2] = sgtol;
    dsave[3] = sxtol;
    dsave[4] = f0;
    dsave[5] = gd;
    dsave[6] = gd0;
    dsave[7] = stp;
    dsave[8] = stpmin;
    dsave[9] = stpmax;
    dsave[10] = scale;
    return 0;
} /* dvmlm_ */

