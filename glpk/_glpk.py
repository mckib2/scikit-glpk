'''GLPK integration.'''

import ctypes
from warnings import warn

import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import OptimizeWarning, OptimizeResult

from ._glpk_defines import GLPK, glp_smcp, glp_iptcp

def glpk(
        c,
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        bounds=None,
        solver='simplex',
        sense=GLPK.GLP_MIN,
        maxit=GLPK.INT_MAX,
        timeout=GLPK.INT_MAX,
        basis_fac='luf+ft',
        message_level=GLPK.GLP_MSG_ERR,
        disp=False,
        simplex_options=None,
        ip_options=None,
        libpath='/home/nicholas/Downloads/glpk-4.65/src/.libs/libglpk.so',
):
    '''GLPK ctypes interface.

    Parameters
    ----------
    c : 1-D array (n,)
        Array of objective coefficients.
    A_ub : 2-D array (m, n)
        scipy.sparse.coo_matrix
    b_ub : 1-D array (m,)
    A_eq : 2-D array (k, n)
        scipy.sparse.coo_matrix
    b_eq : 1-D array (k,)
    bounds : list (n,) of tuple (3,)
        The jth entry in the list corresponds to the jth objective coefficient.
        Each entry is made up of a tuple describing the bounds:

            - ``type`` : {GLP_FR, GLP_LO, GLP_UP, GLP_DB, GLP_FX}
            - ``lb`` : double
            - ``up`` : double

        If the entry is ``None``, then ``type=GLP_FX`` and ``ub==lb==0``.
    solver : { 'simplex', 'interior' }
        Use simplex (LP/MIP) or interior point method (LP only).
        Default is ``simplex``.
    sense : { 'GLP_MIN', 'GLP_MAX' }
        Minimization or maximization problem.
        Default is ``GLP_MIN``.
    scale : bool
        Scale the problem. Default is ``True``.
    maxit : int
        Maximum number of iterations. Default is ``INT_MAX``.
    timout : int
        Limit solution time to ``timeout`` seconds.
        Default is ``INT_MAX``.
    basis_fac : { 'luf+ft', 'luf+cbg', 'luf+cgr', 'btf+cbg', 'btf+cgr' }
        LP basis factorization strategy. Default is ``luf+ft``.
        These are combinations of the following strategies:

            - ``luf`` : plain LU-factorization
            - ``btf`` : block triangular LU-factorization
            - ``ft`` : Forrest-Tomlin update
            - ``cbg`` : Schur complement + Bartels-Golub update
            - ``cgr`` : Schur complement + Givens rotation update

    message_level : { GLP_MSG_OFF, GLP_MSG_ERR, GLP_MSG_ON, GLP_MSG_ON, GLP_MSG_ALL, GLP_MSG_DBG }
        Verbosity level of logging to stdout.
        Only applied when ``disp=True``. Default is ``GLP_MSG_ERR``.
        One of the following:

            ``GLP_MSG_OFF`` : no output
            ``GLP_MSG_ERR`` : warning and error messages only
            ``GLP_MSG_ON`` : normal output
            ``GLP_MSG_ALL`` : full output
            ``GLP_MSG_DBG`` : debug output

    disp : bool
        Display output to stdout. Default is ``False``.
    simplex_options : dict
        Options specific to simplex solver. The dictionary consists of
        the following fields:

            - primal : { 'primal', 'dual', 'dualp' }
                Primal or two-phase dual simplex.
                Default is ``primal``. One of the following:

                    - ``primal`` : use two-phase primal simplex
                    - ``dual`` : use two-phase dual simplex
                    - ``dualp`` : use two-phase dual simplex, and if it fails,
                        switch to the primal simplex

            - init_basis : { 'std', 'adv', 'bib' }
                Choice of initial basis.  Default is 'adv'.
                One of the following:

                    - ``std`` : standard initial basis of all slacks
                    - ``adv`` : advanced initial basis
                    - ``bib`` : Bixby's initial basis

            - steep : bool
                Use steepest edge technique or standard "textbook"
                pricing.  Default is ``True`` (steepest edge).

            - ratio : { 'relax', 'norelax', 'flip' }
                Ratio test strategy. Default is ``relax``.
                One of the following:

                    - ``relax`` : Harris' two-pass ratio test
                    - ``norelax`` : standard "textbook" ratio test
                    - ``flip`` : long-step ratio test

            - presolve : bool
                Use presolver (assumes ``scale=True`` and
                ``init_basis='adv'``. Default is ``True``.

            - exact : bool
                Use simplex method based on exact arithmetic.
                Default is ``False``.

    ip_options : dict
        Options specific to interior-pooint solver.
        The dictionary consists of the following fields:

            - ordering : { 'nord', 'qmd', 'amd', 'symamd' }
                Ordering algorithm used before Cholesky factorizaiton.
                Default is ``amd``. One of the following:

                    - ``nord`` : natural (original) ordering
                    - ``qmd`` : quotient minimum degree ordering
                    - ``amd`` : approximate minimum degree ordering
                    - ``symamd`` : approximate minimum degree ordering
                        algorithm for Cholesky factorization of symmetric matrices.

    mip_options : dict
        Options specific to MIP solver.
        The dictionary consists of the following fields:

            - nomip : bool
                consider all integer variables as continuous
                (allows solving MIP as pure LP). Default is ``False``.
            - branch : { 'first', 'last', 'mostf', 'drtom', 'pcost' }
                Branching rule. Default is ``drtom``.
                One of the following:

                    - ``first`` : branch on first integer variable
                    - ``last`` : branch on last integer variable
                    - ``mostf`` : branch on most fractional variable
                    - ``drtom`` : branch using heuristic by Driebeck and Tomlin
                    - ``pcost`` : branch using hybrid pseudocost heuristic (may be
                        useful for hard instances)

            - backtrack : { 'dfs', 'bfs', 'bestp', 'bestb' }
                Backtracking rule. Default is ``bestb``.
                One of the following:

                    - ``dfs`` : backtrack using depth first search
                    - ``bfs`` : backtrack using breadth first search
                    - ``bestp`` : backtrack using the best projection heuristic
                    - ``bestb`` : backtrack using node with best local bound

            - presolve : bool
                Use MIP presolver. Default is ``True``.

            - binarize : bool
                replace general integer variables by binary ones
                (assumes ``presolve=True``). Default is ``False``.

            - fpump : bool
                Apply feasibility pump heuristic. Default is ``False``.

            - proxy : int
                Apply proximity search heuristic (in seconds). Default is 60.

            - cuts : list of { 'gomory', 'mir', 'cover', 'clique', 'all' }
                Cuts to generate. Default is no cuts. List of the following:

                    - ``gomory`` : Gomory's mixed integer cuts
                    - ``mir`` : MIR (mixed integer rounding) cuts
                    - ``cover`` : mixed cover cuts
                    - ``clique`` : clique cuts
                    - ``all`` : generate all cuts above

            - tol : float
                Relative mip gap tolerance.

            - minisat : bool
                translate integer feasibility problem to CNF-SAT
                and solve it with MiniSat solver. Default is ``False``.

            - bound : float
                add inequality obj <= bound (minimization) or
                obj >= bound (maximization) to integer feasibility
                problem (assumes ``minisat=True``).

    '''

    # Housekeeping
    if bounds is None:
        # Default bound is [0, inf)
        bounds = [(GLPK.GLP_LO, 0, 0)]*len(c)
    if A_ub is None:
        # we need numpy arrays
        A_ub = np.empty((0, len(c)))
    if b_ub is None:
        # just need something iterable
        b_ub = []
    if A_eq is None:
        A_eq = np.empty((0, len(c)))
    if b_eq is None:
        b_eq = []
    if simplex_options is None:
        simplex_options = {}
    if ip_options is None:
        ip_options = {}

    # Get the library
    _lib = GLPK(libpath)._lib

    # Create problem instance
    prob = _lib.glp_create_prob()

    # Give problem a name
    _lib.glp_set_prob_name(prob, b'problem-name')

    # Set objective name
    _lib.glp_set_obj_name(prob, b'obj-name')

    # Set objective sense
    _lib.glp_set_obj_dir(prob, sense)

    # Set objective coefficients and column bounds
    first_col = _lib.glp_add_cols(prob, len(c))
    for ii, (c0, bnd) in enumerate(zip(c, bounds)):
        _lib.glp_set_obj_coef(prob, ii + first_col, c0)
        _lib.glp_set_col_name(prob, ii + first_col, b'c%d' % ii) # name is c[idx], idx is 0-based index

        if bnd is not None:
            _lib.glp_set_col_bnds(prob, ii + first_col, bnd[0], bnd[1], bnd[2])
        # else: default is GLP_FX with lb=0, ub=0

    # Need to load both matrices at the same time
    A = coo_matrix(np.concatenate((A_ub, A_eq), axis=0)) # coo for (i, j, val) format
    b = np.concatenate((b_ub, b_eq))
    first_row = _lib.glp_add_rows(prob, A.shape[0])

    # prepend an element and make 1-based index
    # b/c GLPK expects indices starting at 1
    nnz = A.nnz
    rows = np.concatenate(([-1], A.row + first_row)).astype(ctypes.c_int)
    cols = np.concatenate(([-1], A.col + first_col)).astype(ctypes.c_int)
    values = np.concatenate(([0], A.data)).astype(ctypes.c_double)
    _lib.glp_load_matrix(
        prob,
        nnz,
        rows,
        cols,
        values,
    )

    # Set row bounds
    # Upper bounds (b_ub):
    for ii, b0 in enumerate(b_ub):
        # lb is ignored for upper bounds
        _lib.glp_set_row_bnds(prob, ii + first_row, GLPK.GLP_UP, 0, b0)
    # Equalities (b_eq)
    for ii, b0 in enumerate(b_eq):
        _lib.glp_set_row_bnds(prob, ii + first_row + len(b_ub), GLPK.GLP_FX, b0, b0)

    # Scale the problem
    _lib.glp_scale_prob(prob, GLPK.GLP_SF_AUTO) # do auto scaling for now

    # Select basis factorization method
    bfcp = glp_bfcp()
    _lib.glp_get_bfcp(prob, ctypes.byref(bfcp))
    bfcp.type = {
        'luf+ft': GLPK.GLP_BF_LUF + GLPK.GLP_BF_FT,
        'luf+cbg': GLPK.GLP_BF_LUF + GLPK.GLP_BF_BG,
        'luf+cgr': GLPK.GLP_BF_LUF + GLPK.GLP_BF_GR,
        'btf+cbg': GLPK.GLP_BF_BTF + GLPK.GLP_BF_BG,
        'btf+cgr': GLPK.GLP_BF_BTF + GLPK.GLP_BF_GR,
    }[basis_fac]
    _lib.glp_set_bfcp(prob, ctypes.byref(bfcp))

    # Run the solver
    if solver == 'simplex':

        # Construct an initial basis
        basis = simplex_options.get('init_basis', 'adv')
        basis_fun = {
            'std': _lib.glp_std_basis,
            'adv': _lib.glp_adv_basis,
            'bib': _lib.glp_cpx_basis,
        }[basis]
        basis_args = [prob]
        if basis == 'adv':
            # adv must have 0 as flags argument
            basis_args.append(0)
        basis_fun(*basis_args)

        # Make control structure
        smcp = glp_smcp()
        _lib.glp_init_smcp(ctypes.byref(smcp))

        # Set options
        smcp.msg_lev = message_level*disp
        smcp.meth = {
            'primal': GLPK.GLP_PRIMAL,
            'dual': GLPK.GLP_DUAL,
            'dualp': GLPK.GLP_DUALP,
        }[simplex_options.get('method', 'primal')]
        smcp.pricing = {
            True: GLPK.GLP_PT_PSE,
            False: GLPK.GLP_PT_STD,
        }[simplex_options.get('steep', True)]
        smcp.r_test = {
            'relax': GLPK.GLP_RT_HAR,
            'norelax': GLPK.GLP_RT_STD,
            'flip': GLPK.GLP_RT_FLIP,
        }[simplex_options.get('ratio', 'relax')]
        smcp.it_lim = maxit
        smcp.tm_lim = timeout
        smcp.presolve = {
            True: GLPK.GLP_ON,
            False: GLPK.GLP_OFF,
        }[simplex_options.get('presolve', GLPK.GLP_OFF)]

        # Simplex driver
        if simplex_options.get('exact', False):
            ret_code = _lib.glp_exact(prob, ctypes.byref(smcp))
        else:
            ret_code = _lib.glp_simplex(prob, ctypes.byref(smcp))
        if ret_code != GLPK.SUCCESS:
            warn('GLPK simplex not successful!', OptimizeWarning)
            return OptimizeResult({
                'message': GLPK.RET_CODES[ret_code],
            })

        # Figure out what happened
        status = _lib.glp_get_status(prob)
        message = GLPK.STATUS_CODES[status]
        res = OptimizeResult({
            'status': status,
            'message': message,
            'success': status == GLPK.GLP_OPT,
        })

        # We can read a solution:
        if status == GLPK.GLP_OPT:

            res.fun = _lib.glp_get_obj_val(prob)
            res.x = np.array([_lib.glp_get_col_prim(prob, ii) for ii in range(1, len(c)+1)])
            res.dual = np.array([_lib.glp_get_col_dual(prob, ii) for ii in range(1, len(b_ub)+1)])

            # We don't get slack without doing sensitivity analysis since GLPK uses
            # auxiliary variables instead of slack!
            res.slack = b_ub - A_ub @ res.x
            res.con = b_eq - A_eq @ res.x

            # We shouldn't be reading this field... But we will anyways
            res.nit = prob.contents.it_cnt

    elif solver == 'interior':

        # Make a control structure
        iptcp = glp_iptcp()
        _lib.glp_init_iptcp(ctypes.byref(iptcp))

        # Set options
        iptcp.msg_lev = message_level*disp
        iptcp.ord_alg = {
            'nord': GLPK.GLP_ORD_NONE,
            'qmd': GLPK.GLP_ORD_QMD,
            'amd': GLPK.GLP_ORD_AMD,
            'symamd': GLPK.GLP_ORD_SYMAMD,
        }[ip_options.get('ordering', 'amd')]

        # Run the solver
        ret_code = _lib.glp_interior(prob, ctypes.byref(iptcp))
        if ret_code != GLPK.SUCCESS:
            warn('GLPK interior-point not successful!', OptimizeWarning)
            return OptimizeResult({
                'message': GLPK.RET_CODES[ret_code],
            })

        # Figure out what happened
        status = _lib.glp_ipt_status(prob)
        message = GLPK.STATUS_CODES[status]
        res = OptimizeResult({
            'status': status,
            'message': message,
            'success': status == GLPK.GLP_OPT,
        })

        # We can read a solution:
        if status == GLPK.GLP_OPT:

            res.fun = _lib.glp_ipt_obj_val(prob)
            res.x = np.array([_lib.glp_ipt_col_prim(prob, ii) for ii in range(1, len(c)+1)])
            res.dual = np.array([_lib.glp_ipt_col_dual(prob, ii) for ii in range(1, len(b_ub)+1)])

            # We don't get slack without doing sensitivity analysis since GLPK uses
            # auxiliary variables instead of slack!
            res.slack = b_ub - A_ub @ res.x
            res.con = b_eq - A_eq @ res.x

            # We shouldn't be reading this field... But we will anyways
            res.nit = prob.contents.it_cnt

    else:
        raise ValueError('"%s" is not a recognized solver.' % solver)

    # We're done, cleanup!
    _lib.glp_delete_prob(prob)

    # Map status codes to scipy:
    res.status = {
        GLPK.GLP_OPT: 0,
    }[res.status]

    return res

if __name__ == '__main__':
    pass
