/* dlmmv.f -- translated by f2c (version 20200916).
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

/* Subroutine */ int dlmmv_(integer *n, integer *m, doublereal *s, doublereal 
	*y, doublereal *rho, doublereal *scale, integer *mark, doublereal *v, 
	doublereal *wa)
{
    /* System generated locals */
    integer s_dim1, s_offset, y_dim1, y_offset, i__1;
    doublereal d__1;

    /* Local variables */
    static integer i__, k;
    static doublereal beta;
    extern doublereal ddot_(integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *), daxpy_(integer *, doublereal *, doublereal *, integer 
	    *, doublereal *, integer *);

/*     ********** */

/*     This subroutine computes the matrix-vector product H*v */
/*     where H is the inverse BFGS approximation. */

/*     The matrix H depends on an initial matrix H0, m steps */
/*     s(1),...,s(m), and m gradient differences y(1),...,y(m). */
/*     These vectors are stored in the arrays s and y. */
/*     The most recent step and gradient difference are stored */
/*     in columns s(1,mark) and y(1,mark), respectively. */
/*     The initial matrix H0 is assumed to be scale*I. */

/*     The subroutine statement is */

/*       subroutine dlmmv(n,m,s,y,rho,scale,mark,v,wa) */

/*     where */

/*       n is an integer variable. */
/*         On entry n is the number of variables. */
/*         On exit n is unchanged. */

/*       m is an integer variable. */
/*         On entry m specifies the number of steps and gradient */
/*            differences that are stored. */
/*         On exit m is unchanged. */

/*       s is a double precision array of dimension (n,m) */
/*         On entry s contains the m steps. */
/*         On exit s is unchanged. */

/*       y is a double precision array of dimension (n,m) */
/*         On entry y contains the m gradient differences. */
/*         On exit y is unchanged. */

/*       rho is a double precision array of dimension m */
/*         On entry rho contains the m innerproducts (s(i),y(i)). */
/*         On exit rho is unchanged. */

/*       scale is a double precision variable */
/*         On entry scale specifies the initial matrix H0 = scale*I. */
/*         On exit scale is unchanged. */

/*       mark is an integer variable. */
/*         On entry mark points to the current s(i) and y(i). */
/*         On exit mark is unchanged. */

/*       v is a double precision array of dimension n. */
/*         On entry v contains the vector v. */
/*         On exit v contains the matrix-vector product H*v. */

/*       wa is a double precision work array of dimension m. */

/*     Subprograms called */

/*       Level 1 BLAS ... daxpy, ddot, dscal */

/*     MINPACK-2 project. November 1993. */
/*     Argonne National Laboratory and University of Minnesota. */
/*     Brett M. Averick, Richard G. Carter, and Jorge J. More'. */

/*     ********** */
    /* Parameter adjustments */
    --v;
    --wa;
    --rho;
    y_dim1 = *n;
    y_offset = 1 + y_dim1;
    y -= y_offset;
    s_dim1 = *n;
    s_offset = 1 + s_dim1;
    s -= s_offset;

    /* Function Body */
    k = *mark + 1;
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	--k;
	if (k == 0) {
	    k = *m;
	}
	wa[k] = ddot_(n, &s[k * s_dim1 + 1], &c__1, &v[1], &c__1) / rho[k];
	d__1 = -wa[k];
	daxpy_(n, &d__1, &y[k * y_dim1 + 1], &c__1, &v[1], &c__1);
/* L10: */
    }
    dscal_(n, scale, &v[1], &c__1);
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	beta = wa[k] - ddot_(n, &y[k * y_dim1 + 1], &c__1, &v[1], &c__1) / 
		rho[k];
	daxpy_(n, &beta, &s[k * s_dim1 + 1], &c__1, &v[1], &c__1);
	++k;
	if (k == *m + 1) {
	    k = 1;
	}
/* L20: */
    }
    return 0;
} /* dlmmv_ */

