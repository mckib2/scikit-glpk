sckit-glpk
----------

Proof of concept Python wrappers for GLPK.

Installation
------------

Should be an easy pip installation:

.. code-block::

   pip install scikit-glpk

Usage
-----

There are a few things in this package:

- `glpk()` : the wrappers over the solvers (basically acts like Pythonized `glpsol`)
- `mpsread()` : convert an MPS file to some matrices



.. code-block::

   from glpk import glpk, GLPK

   res = glpk(
       c, A_ub, b_ub, A_eq, b_eq, bounds, solver, sense, maxit, timeout,
       basis_fac, message_level, disp, simplex_options, ip_options,
       mip_options, libpath)

   c, A_ub, b_ub, A_eq, b_eq, bounds = mpsread(
       filename, fmt=GLPK.GLP_MPS_FILE, ret_glp_prob=False, libpath=None)

There's lots of information in the docstrings for these functions, please check there for a complete listing and explanation.

Notice that `glpk` is the wrapper and `GLPK` acts as a namespace that holds constants.

`bounds` is also behaves a little differently for both of these:

- as an input to `glpk`, `bounds` is a list of triplets in the style of GLPK (probably should be converted to `linprog` conventions)
- as an output of `mpsread`, `bounds` is a list of tuples in the style of `linprog`

GLPK stuffs
-----------

GLPK must be installed in order to use the wrappers. Download the latest version from `here <http://ftp.gnu.org/gnu/glpk/>`_ and follow the instructions for installation.  If you use Linux/Mac, you should be able to run the following to compile from source (see docs for different configuration options):

.. code-block::

   ./configure
   make -j
   make install

For Windows you will need at least `Visual Studio Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_.  Go to the correct subdirectory (w32 for 32-bit or w64 for 64-bit) and the run the batch script:

.. code-block::

   Build_GLPK_with_VC14_DLL.bat

To use the GLPK installation, either provide the location of the shared library to the function, i.e. `glpk(..., libpath='path/to/libglpk.so')` or set the environment variable `GLPK_LIB_PATH=path/to/libglpk.so`.  The wrappers have nothing to wrap if they don't know where to find the library.

If you already have `Octave <https://www.gnu.org/software/octave/>`_ installed, note that GLPK is bundled with the installation, so you can find `libglpk.[so|dll]` in the `bin` directory and do not have to install it from source as above.

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
- `pulp <https://launchpad.net/pulp-or>`_ : Calls `glpsol` from command line (writes problems, solutions to file instead of shared memory model -- this is actually easy to do)
- `yaposib <https://github.com/coin-or/yaposib>`_ : seems dead? OSI-centric
- `ecyglpki <https://github.com/equaeghe/ecyglpki/tree/0.1.0>`_ : GPL licensed, dead?
- `swiglpk <https://github.com/biosustain/swiglpk>`_ : GPL licensed, low level
- `optlang <https://github.com/biosustain/optlang>`_ : sympy-like, cool project otherwise

Most existing projects lean to GPL licenses.  Not a bad thing, but would hinder adoption into scipy.

Why do we want this?
--------------------

GLPK has a lot of options that the current scipy solvers lack as well as robust MIP support (only basic in HiGHS).  It is also a standard, well known solver in the optimization community.  The only thing that I want that it lacks on an API level is robust support for column generation.  Easy access to GLPK as a backend to `linprog` would be very welcome (to me at least).

Approach
--------

Since the underlying API is quite simple and written in C and only C, `ctypes` is a good fit for this.

GLPK will not be packaged with scipy due to licensing issues, so the strategy will be to specify where the installation is on a user's computer (i.e., path to the shared library).  `linprog` could then presumably route the problem to the GLPK backend instead of HiGHS or the existing native python solvers.

The `ctypes` wrapper is required for integrating GLPK into the Python runtime.  Instead of using MPS files to communicate problems and reading solutions from files, `scipy.sparse.coo_matrix` and `numpy` arrays can be passed directly to the library.  More information can be extracted from GLPK this way as well (For example, there is no way to get iteration count except by reading directly from the underlying structs.  It is only ever printed to stdout, no other way to get it).

TODO
----

- Several GLPK solver options (notably tolerances) not wrapped yet
