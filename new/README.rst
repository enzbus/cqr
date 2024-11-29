CQR: Conic QR Solver
====================

Experimental solver for convex conic programs based on the QR decomposition.

Installation
------------

.. code-block::

	pip install cqr

Usage
-----

From CVXPY:

.. code-block:: python
	
	from cqr import CQR
	import cvxpy
	
	cvxpy.Problem(...).solve(solver=CQR())

