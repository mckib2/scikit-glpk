'''GLPK integration.'''

import ctypes

import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import OptimizeWarning, OptimizeResult

from ._glpk_defines import GLPK

def _glpk(
        c,
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        bounds=None,
        sense=GLPK.GLP_MIN,
        solver='simplex',
        maxit=GLPK.INT_MAX,
        timeout=GLPK.INT_MAX,
        simplex_options=None,
        message_level=GLPK.GLP_MSG_OFF,
        disp=False,
):
    '''GLPK ctypes interface.

    Parameters
    ----------
    c : 1-D array (n,)
        Array of objective coefficients.
    bounds : list (n,) of tuple (3,)
        The jth entry in the list corresponds to the jth objective coefficient.
        Each entry is made up of a tuple describing the bounds:

            - ``type`` : {GLP_FR, GLP_LO, GLP_UP, GLP_DB, GLP_FX}
            - ``lb`` : double
            - ``up`` : double

        If the entry is ``None``, then ``type=GLP_FX`` and ``ub==lb==0``.
    sense : {GLP_MIN, GLP_MAX}
        Minimize or maxiximize. Default is ``GLP_MIN``.
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

    # Get the library
    glpk = GLPK._lib

    # Create problem instance
    prob = glpk.glp_create_prob()

    # Give problem a name
    glpk.glp_set_prob_name(prob, b'problem-name')

    # Set objective name
    glpk.glp_set_obj_name(prob, b'obj-name')

    # Set objective sense
    glpk.glp_set_obj_dir(prob, sense)

    # Set objective coefficients and column bounds
    first_col = glpk.glp_add_cols(prob, len(c))
    for ii, (c0, bnd) in enumerate(zip(c, bounds)):
        glpk.glp_set_obj_coef(prob, ii + first_col, c0)
        glpk.glp_set_col_name(prob, ii + first_col, b'c%d' % ii) # name is c[idx], idx is 0-based index

        if bnd is not None:
            glpk.glp_set_col_bnds(prob, ii + first_col, bnd[0], bnd[1], bnd[2])
        # else: default is GLP_FX with lb=0, ub=0

    # Need to load both matrices at the same time
    A = coo_matrix(np.concatenate((A_ub, A_eq), axis=0)) # coo for (i, j, val) format
    b = np.concatenate((b_ub, b_eq))
    first_row = glpk.glp_add_rows(prob, A.shape[0])

    # prepend an element and make 1-based index
    # b/c GLPK expects indices starting at 1
    nnz = A.nnz
    rows = np.concatenate(([-1], A.row + first_row)).astype(ctypes.c_int)
    cols = np.concatenate(([-1], A.col + first_col)).astype(ctypes.c_int)
    values = np.concatenate(([0], A.data)).astype(ctypes.c_double)
    glpk.glp_load_matrix(
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
        glpk.glp_set_row_bnds(prob, ii + first_row, GLPK.GLP_UP, 0, b0)
    # Equalities (b_eq)
    for ii, b0 in enumerate(b_eq):
        glpk.glp_set_row_bnds(prob, ii + first_row + len(b_ub), GLPK.GLP_FX, b0, b0)

    # Scale the problem
    glpk.glp_scale_prob(prob, GLPK.GLP_SF_AUTO) # do auto scaling for now

    # Run the solver
    if solver == 'simplex':

        # Construct an initial basis
        basis = simplex_options.get('basis', 'adv')
        basis_fun = {
            'std': glpk.glp_std_basis,
            'adv': glpk.glp_adv_basis,
            'bib': glpk.glp_cpx_basis,
        }[basis]
        basis_args = [prob]
        if basis == 'adv':
            # adv must have 0 as flags argument
            basis_args.append(0)
        basis_fun(*basis_args)

        # Make control structure
        smcp = GLPK.glp_smcp()
        glpk.glp_init_smcp(ctypes.byref(smcp))

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
        ret_code = glpk.glp_simplex(prob, ctypes.byref(smcp))
        if ret_code != GLPK.SUCCESS:
            warn('GLPK simplex not successful!', OptimizeWarning)
            return OptimizeResult({
                'message': GLPK.RET_CODES[ret_code],
            })

        # Figure out what happened
        status = glpk.glp_get_status(prob)
        message = GLPK.STATUS_CODES[status]
        res = OptimizeResult({
            'status': status,
            'message': message,
            'success': status == GLPK.GLP_OPT,
        })

        # We can read a solution:
        if status == GLPK.GLP_OPT:

            res.fun = glpk.glp_get_obj_val(prob)
            res.x = np.array([glpk.glp_get_col_prim(prob, ii) for ii in range(1, len(c)+1)])
            res.dual = np.array([glpk.glp_get_col_dual(prob, ii) for ii in range(1, len(b_ub)+1)])

            # We don't get slack without doing sensitivity analysis since GLPK uses
            # auxiliary variables instead of slack!
            res.slack = b_ub - A_ub @ res.x
            res.con = b_eq - A_eq @ res.x

            # We shouldn't be reading this field... But we will anyways
            res.nit = prob.contents.it_cnt

    else:
        raise NotImplementedError()

    # We're done, cleanup!
    glpk.glp_delete_prob(prob)

    # Map status codes to scipy:
    res.status = {
        GLPK.GLP_OPT: 0,
    }[res.status]

    return res

if __name__ == '__main__':
    pass
