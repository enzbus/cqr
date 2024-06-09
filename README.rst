Project EuroMir
===============

Project codename `EuroMir <https://rcdb.com/972.htm>`_
`(also, song) <https://open.spotify.com/track/3ffkbz5OvPjXjOsYTsEjKu>`_
is a prototype of a conic programming solver that builds on the work done for
the older `conic program refinement
<https://github.com/cvxgrp/cone_prog_refine>`_ research project, of which it
may inherit the name, once it's completed.

Compared to that 2018 Stanford research project, it has a completely new
codebase (written from scratch) and it removes various unnecessary
dependencies. It also has a modified algorithm, which is guaranteed to preserve
convexity (unlike many similar attempts). It uses a simplified version
of the 2018 algorithm only for the final polishing.

Algorithm (draft)
=================

The algorithm is under development. This is the current model, see the
`scs <https://web.stanford.edu/~boyd/papers/pdf/scs.pdf>`_ and
`conic refinement
<https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf>`_ papers to
better understand the notation:

.. math::

    \begin{array}{ll}

        \text{minimize} & \|Q u - v \|_2^2 + \| u - \Pi u \|_2^2  + \| v - \Pi^\star v \|_2^2

    \end{array}

This program always has a non-zero solution for which the objective is zero,
thanks to the guarantees from the convex duality theory of the `homogeneous
self-dual embedding <https://doi.org/10.1287/moor.19.1.53>`_.
The system matrix :math:`Q` is skew symmetric, so at convergence it is
guaranteed that :math:`u` and :math:`v` are orthogonal, and hence no other
requirements are needed on the formulation above to recover an optimal solution
(or certificate) for the original program.

The objective function is clearly convex and has continuous derivative. The
conditioning depends on the conditioning of :math:`Q`, we apply by default
standard `Ruiz diagonal pre-conditioning
<https://web.stanford.edu/~takapoui/preconditioning.pdf>`_.

We use `limited-memory BFGS
<https://doi.org/10.1090/S0025-5718-1980-0572855-7>`_, as it's implemented in
the `variable-metric limited memory module of MINPACK-2
<https://ftp.mcs.anl.gov/pub/MINPACK-2/vmlm>`_, for the minimization routine.

We then use the 2018 conic refinement algorithm (simplified, without the
normalization step), using `LSQR
<https://web.stanford.edu/group/SOL/software/lsqr/>`_ on the HDSE residual
operator, for the final refinement of the approximate solution obtained from
the BFGS loop.

The approach is clearly globally convergent, and it can probably be showed to
converge super-linearly. However, theoretical bounds on convergence have very
limited applicability in real mathematical engineering; smart implementation
choices and good code quality are by far more important. Only experimentation
can tell how this approach compares to mature alternative ones, such as
the various flavors of interior point and operator splitting algorithms.


Development
===========

We mirror development in Python and C (and, in the future, C with OpenMP). We
set up the build with CMake, patching ``setuptools`` to use it. All testing and
profiling is done in Python. Binding of C to Python is done using ``ctypes``:
we link at runtime. This guarantees that pre-built wheels are agnostic to the
Python version, the Numpy ABI, ....
Memory communication between Python (mostly Numpy arrays) and C is still
zero-copy and the GIL is released by ``ctypes``; there are no appreciable
overheads compared to building with ``Python.h``, since the user only calls a
``ctypes``-linked function once when solving a conic program.

Memory
======

All memory is pre-allocated by the caller, which in typical usage is the CVXPY
interface. There are no ``malloc`` in our C code. We require to allocate space
for a copy of the problem data (for rescaling, that may be prevented if we
change the internal logic, adding some computational overhead), and the size of
the primal and dual variables times the L-BFGS memory, which is a small number
like 5 or 10.

In layman terms, this is the least memory any conic solver needs, and
dramatically less than interior-point solvers.

Usage
=====

We will provide a CVXPY and raw Python interface as part of our packages. The
single C function the user interacts with will be also documented, for usage
from other runtime environments.

Installation
============

Pre-built wheels will be available on PyPi soon.
