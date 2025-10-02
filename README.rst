CQR: Conic QR Solver
====================

Experimental solver for convex conic programs based on the QR decomposition.

**Work in progress**, not ready for use.

(Pre-alpha) installation
------------------------

.. code-block::

	pip install cqr

Usage
-----

From CVXPY < 1.7:

.. code-block:: python
	
	from cqr import CQR
	import cvxpy
	
	cvxpy.Problem(...).solve(solver=CQR())

