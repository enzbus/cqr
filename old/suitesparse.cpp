// sudo apt install libsuitesparse-dev
// g++ suitesparse.cpp -I/usr/include/suitesparse -lcholmod -lspqr -Wconversion -Wall
//
// below it's working householder QR decomp
//
// from https://stackoverflow.com/questions/6075442/creating-a-sparse-matrix-in-cholmod-or-suitesparseqr
// and SuiteSparseQR.hpp, line 502 (GH version Oct 2024)
//
#include "SuiteSparseQR.hpp"
#include "SuiteSparse_config.h"

int main (int argc, char **argv)
{
  cholmod_common Common, *cc;
  cholmod_sparse *A;
  cholmod_dense *X, *B;

  // start CHOLMOD
  cc = &Common;
  cholmod_l_start (cc);

  /* A =
      [
        1.1,  0.0, -0.5,  0.7
        0.0, -2.0,  0.0,  0.0
        0.0,  3.0,  0.9,  0.0
        0.0,  0.0,  0.0,  0.6
      ]
  */
  int m = 4;   // num rows in A
  int n = 4;   // num cols in A
  int nnz = 7; // num non-zero elements in A
  int unsymmetric = 0; // A is non-symmetric: see cholmod.h > search for `stype` for more details

  // In coordinate form (COO) a.k.a. triplet form (zero-based indexing)
  int i[nnz] = {0, 1, 0, 2, 2, 0, 3}; // row indices
  int j[nnz] = {0, 1, 2, 1, 2, 3, 3}; // col indices
  double x[nnz] = {1.1, -2.0, -0.5, 3., 0.9, 0.7, 0.6}; // values

  // Set up the cholmod matrix in COO/triplet form
  cholmod_triplet *T = cholmod_l_allocate_triplet(m, n, nnz, unsymmetric, CHOLMOD_REAL, cc);
  T->nnz = nnz;

  for (int ind = 0; ind < nnz; ind++)
  {
    ((long int *) T->i)[ind] = i[ind];     // Notes:
    ((long int *) T->j)[ind] = j[ind];     // (1) casting necessary because these are void* (see cholmod.h)
    ((double *) T->x)[ind] = x[ind];       // (2) direct assignment will cause memory corruption
  }                                        // (3) long int for index pointers corresponds to usage of cholmod_l_* functions

  // convert COO/triplet to CSC (compressed sparse column) format
  A = (cholmod_sparse *) cholmod_l_triplet_to_sparse(T, nnz, cc);
  // note: if you already know CSC format you can skip the triplet allocation and instead use cholmod_allocate_sparse
  //       and assign the member variables: see cholmod.h > cholmod_sparse_struct definition

  cholmod_l_print_sparse(A, "matrice A", cc);

//
// QR decomp
//

  // [Q,R,E] = qr(A) where Q is returned in Householder form
int64_t something;
cholmod_sparse *R;
cholmod_sparse *H;
cholmod_dense * HTau;
int64_t * perm;
int64_t * perm1;
something = SuiteSparseQR <double>
(
    // inputs, not modified
    (long int) 3, //int ordering,           // all, except 3:given treated as 0:fixed
    0., //double tol,             // only accept singletons above tol
    (long int) m, //Int econ,  // number of rows of C and R to return
    A,      // m-by-n sparse matrix
    // outputs, allocated here
    &R,     // the R factor
    &perm,   // permutation of 0:n-1, NULL if identity
    &H,     // the Householder vectors (m-by-nh)
    &perm1,// size m; row permutation for H
    &HTau,   // size nh, Householder coefficients
    cc      // workspace and parameters
) ;

cholmod_l_print_sparse(R, "matrice R", cc);
cholmod_l_print_sparse(H, "matrice H", cc);
cholmod_l_print_dense(HTau, "matrice HTau", cc);




  // B = ones (size (A,1),1)
  B = cholmod_l_ones (A->nrow, 1, A->xtype, cc);

  // X = A\B
  X = SuiteSparseQR <double> (A, B, cc);

  // Print contents of X
  printf("X = [\n");
  for (int ind = 0; ind < n; ind++)
  {
    printf("%f\n", ((double *) X->x)[ind]);
  }
  printf("]\n");
  fflush(stdout);

  // free everything and finish CHOLMOD
  cholmod_l_free_triplet (&T, cc);
  cholmod_l_free_sparse (&A, cc);
  cholmod_l_free_dense (&X, cc);
  cholmod_l_free_dense (&B, cc);
  cholmod_l_finish (cc);
  return 0;
}
