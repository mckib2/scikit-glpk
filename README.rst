sckit-glpk
----------

Proof of concept Python wrappers for GLPK.

Installation
------------

Should be an easy pip installation:

.. code-block::

   pip install scikit-glpk

GLPK must be installed in order to use the wrappers.

Background
----------

The `GNU Linear Programming Kit (GLPK) <https://www.gnu.org/software/glpk/>`_ has simplex, interior-point, and MIP solvers all callable from a C library.  We would like to be able to use these from within Python and be potentially included as a backend for scipy's `linprog` function.

Note that there are several projects that aim for something like this, but which don't match up for what I'm looking for:

- `python-glpk <https://www.dcc.fc.up.pt/~jpp/code/python-glpk/>`_ : no longer maintained (dead as of ~2013)
- `PyGLPK <http://tfinley.net/software/pyglpk/>`_ : GPL licensed
- `PyMathProg <https://pypi.org/project/pymprog/>`_ : GPL licensed, uses different conventions than that of `linprog`
- `Pyomo <https://github.com/Pyomo/pyomo>`_ : Big, uses different conventions than that of `linprog`
- `CVXOPT <https://cvxopt.org/>`_ : Big, GPL licensed
- `Sage <https://git.sagemath.org/sage.git/tree/README.md>`_ : Big, GPL licensed
- `pulp <https://launchpad.net/pulp-or>`_ : Calls `glpsol` from command line (writes problems, solutions to file instead of shared memory model)
- `yaposib <https://github.com/coin-or/yaposib>`_ : seems dead? Bigger than I want
- `ecyglpki <https://github.com/equaeghe/ecyglpki/tree/0.1.0>`_ : dead
- `swiglpk <https://github.com/biosustain/swiglpk>`_ : GPL licensed, low level

Most existing projects lean to GPL licenses.  Not a bad thing, but would hinder adoption into scipy.

Why do we want this?
--------------------

GLPK has a lot of options that the current scipy solvers lack as well as robust MIP support (only basic in HiGHS).  It is also a standard, well known solver in the optimization community.  Easy access to GLPK as a backend to `linprog` would be very welcome (to me at least).

Approach
--------

Since the underlying API is quite simple and written in C, `ctypes` is a good fit for this.

GLPK will not be packaged with scipy due to licensing issues, so the strategy will be to specify where the installation is on a user's computer (i.e., path to the shared library).  `linprog` could then presumably route the problem to the GLPK backend instead of HiGHS or the existing native python solvers.

The `ctypes` wrapper is required for integrating GLPK into the Python runtime.  Instead of using MPS files to communicate problems and reading solutions from files, `scipy.sparse.coo_matrix` and `numpy` arrays can be passed directly to the library.  More information can be extracted from GLPK this way as well (For example, there is no way to get iteration count except by reading directly from the underlying structs.  It is only ever printed to stdout, no other way to get it).
