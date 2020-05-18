
import ctypes

from numpy.ctypeslib import ndpointer

class GLPK:

    INT_MAX = ctypes.c_uint(-1).value // 2
    GLP_ON = 1
    GLP_OFF = 0

    # solution status:
    GLP_UNDEF = 1   # solution is undefined
    GLP_FEAS = 2    # solution is feasible
    GLP_INFEAS = 3  # solution is infeasible
    GLP_NOFEAS = 4  # no feasible solution exists
    GLP_OPT = 5     # solution is optimal
    GLP_UNBND = 6   # solution is unbounded
    STATUS_CODES = {
        GLP_UNDEF: 'solution is undefined',
        GLP_FEAS: 'solution is feasible',
        GLP_INFEAS: 'solution is infeasible',
        GLP_NOFEAS: 'no feasible solution exists',
        GLP_OPT: 'solution is optimal',
        GLP_UNBND: 'solution is unbounded',
    }

    # Load the shared library
    so = '/home/nicholas/Downloads/glpk-4.65/src/.libs/libglpk.so'
    _lib = ctypes.cdll.LoadLibrary(so)
    _lib.glp_version.restype = ctypes.c_char_p

    class glp_prob(ctypes.Structure):
        class DMP(ctypes.Structure):
            _fields_ = []
        class glp_tree(ctypes.Structure):
            _fields_ = []
        class GLPROW(ctypes.Structure):
            _fields_ = []
        class GLPCOL(ctypes.Structure):
            _fields_ = []
        class AVL(ctypes.Structure):
            _fields_ = []
        class BFD(ctypes.Structure):
            _fields_ = []
        # shouldn't access these directly!
        _fields_ = [
            ('pool', ctypes.POINTER(DMP)),
            ('tree', ctypes.POINTER(glp_tree)),
            ('name', ctypes.c_char_p),
            ('obj', ctypes.c_char_p),
            ('dir', ctypes.c_int),
            ('c0', ctypes.c_double),
            ('m_max', ctypes.c_int),
            ('n_max', ctypes.c_int),
            ('m', ctypes.c_int),
            ('n', ctypes.c_int),
            ('nnz', ctypes.c_int),
            ('row', ctypes.POINTER(ctypes.POINTER(GLPROW))),
            ('col', ctypes.POINTER(ctypes.POINTER(GLPCOL))),
            ('r_tree', ctypes.POINTER(AVL)),
            ('c_tree', ctypes.POINTER(AVL)),
            ('valid', ctypes.c_int),
            ('head', ctypes.POINTER(ctypes.c_int)),
            ('bfd', ctypes.POINTER(BFD)),
            ('pbs_stat', ctypes.c_int),
            ('dbs_stat', ctypes.c_int),
            ('obj_val', ctypes.c_double),
            ('it_cnt', ctypes.c_int),
            ('some', ctypes.c_int),
            ('ipt_stat', ctypes.c_int),
            ('ipt_obj', ctypes.c_double),
            ('mip_stat', ctypes.c_int),
            ('mip_obj', ctypes.c_double),
        ]

    _lib.glp_create_prob.restype = ctypes.POINTER(glp_prob)

    # Setters
    _lib.glp_set_prob_name.restype = None
    _lib.glp_set_prob_name.argtypes = [ctypes.POINTER(glp_prob), ctypes.c_char_p]

    _lib.glp_set_obj_name.restype = None
    _lib.glp_set_obj_name.argtypes = [ctypes.POINTER(glp_prob), ctypes.c_char_p]

    _lib.glp_set_obj_dir.restype = None
    _lib.glp_set_obj_dir.argtypes = [ctypes.POINTER(glp_prob), ctypes.c_int]

    _lib.glp_add_cols.restype = ctypes.c_int
    _lib.glp_add_cols.argtypes = [ctypes.POINTER(glp_prob), ctypes.c_int]

    _lib.glp_set_obj_coef.restype = None
    _lib.glp_set_obj_coef.argtypes = [ctypes.POINTER(glp_prob), ctypes.c_int, ctypes.c_double]

    _lib.glp_set_col_name.restype = None
    _lib.glp_set_col_name.argtypes = [ctypes.POINTER(glp_prob), ctypes.c_int, ctypes.c_char_p]

    _lib.glp_set_col_bnds.restype = None
    _lib.glp_set_col_bnds.argtypes = [
        ctypes.POINTER(glp_prob),
        ctypes.c_int,    # col index (1-based)
        ctypes.c_int,    # type
        ctypes.c_double, # lb
        ctypes.c_double, # up
    ]

    _lib.glp_add_rows.restype = ctypes.c_int
    _lib.glp_add_rows.argtypes = [ctypes.POINTER(glp_prob), ctypes.c_int]

    _lib.glp_load_matrix.restype = None
    _lib.glp_load_matrix.argtypes = [
        ctypes.POINTER(glp_prob),
        ctypes.c_int,                                     # nnz
        ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'),    # row indices
        ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'),    # col indices
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), # values
    ]

    _lib.glp_set_row_bnds.restype = None
    _lib.glp_set_row_bnds.argtypes = [
        ctypes.POINTER(glp_prob),
        ctypes.c_int,    # row index (1-based)
        ctypes.c_int,    # type
        ctypes.c_double, # lb
        ctypes.c_double, # ub
    ]

    _lib.glp_scale_prob.restype = None
    _lib.glp_scale_prob.argtypes = [ctypes.POINTER(glp_prob), ctypes.c_int]

    _lib.glp_std_basis.restype = None
    _lib.glp_std_basis.argtypes = [ctypes.POINTER(glp_prob)]

    _lib.glp_adv_basis.restype = None
    _lib.glp_adv_basis.argtypes = [ctypes.POINTER(glp_prob), ctypes.c_int]

    _lib.glp_cpx_basis.restype = None
    _lib.glp_cpx_basis.argtypes = [ctypes.POINTER(glp_prob)]


    # Getters
    _lib.glp_get_prob_name.restype = ctypes.c_char_p
    _lib.glp_get_prob_name.argtypes = [ctypes.POINTER(glp_prob)]

    _lib.glp_get_obj_name.restype = ctypes.c_char_p
    _lib.glp_get_obj_name.argtypes = [ctypes.POINTER(glp_prob)]

    _lib.glp_get_obj_dir.restype = ctypes.c_int
    _lib.glp_get_obj_dir.argtypes = [ctypes.POINTER(glp_prob)]

    _lib.glp_get_obj_coef.restype = ctypes.c_double
    _lib.glp_get_obj_coef.argtypes = [ctypes.POINTER(glp_prob), ctypes.c_int]

    _lib.glp_get_col_name.restype = ctypes.c_char_p
    _lib.glp_get_col_name.argtypes = [ctypes.POINTER(glp_prob), ctypes.c_int]

    _lib.glp_get_num_rows.restype = ctypes.c_int
    _lib.glp_get_num_rows.argtypes = [ctypes.POINTER(glp_prob)]

    _lib.glp_get_num_nz.restype = ctypes.c_int
    _lib.glp_get_num_nz.argtypes = [ctypes.POINTER(glp_prob)]

    _lib.glp_get_status.restype = ctypes.c_int
    _lib.glp_get_status.argtypes = [ctypes.POINTER(glp_prob)]

    _lib.glp_get_obj_val.restype = ctypes.c_double
    _lib.glp_get_obj_val.argtypes = [ctypes.POINTER(glp_prob)]

    _lib.glp_get_col_prim.restype = ctypes.c_double
    _lib.glp_get_col_prim.argtypes = [
        ctypes.POINTER(glp_prob),
        ctypes.c_int, # primal value of jth col
    ]

    _lib.glp_get_col_dual.restype = ctypes.c_double
    _lib.glp_get_col_dual.argtypes = [
        ctypes.POINTER(glp_prob),
        ctypes.c_int, # dual value of jth col
    ]

    # Pricing techniques
    GLP_PT_STD = 17 # 0x11 # standard (Dantzig's rule)
    GLP_PT_PSE = 34 # 0x22 # projected steepest edge

    # Ratio test techniques
    GLP_RT_STD = 17 # 0x11
    GLP_RT_HAR = 34 # 0x22
    GLP_RT_FLIP = 51 # 0x33

    # Objective sense
    GLP_MIN = 1
    GLP_MAX = 2

    # Bound types
    GLP_FR = 1 # -inf <  x  < +inf
    GLP_LO = 2 #   lb <= x  < +inf
    GLP_UP = 3 # -inf <  x <= ub
    GLP_DB = 4 #   lb <= x <= ub
    GLP_FX = 5 #   lb == x == ub

    # Scaling techniques
    GLP_SF_GM = 1     # 0x01   # geometric scaling
    GLP_SF_EQ = 16    # 0x10   # equilibration scaling
    GLP_SF_2N = 3     # 0x20   # round scale factors to nearest power of two
    GLP_SF_SKIP = 64  # 0x40 # skip scaling, if the problem is well scaled
    GLP_SF_AUTO = 128 # 0x80 # choose scaling options automatically

    # return codes for glp_simplex driver
    SUCCESS = 0
    GLP_EBADB = 1
    GLP_ESING = 2
    GLP_ECOND = 3
    GLP_EBOUND = 4
    GLP_EFAIL = 5
    GLP_EOBJLL = 6
    GLP_EOBJUL = 7
    GLP_EITLIM = 8
    GLP_ETMLIM = 9
    GLP_ENOPFS = 10
    GLP_ENODFS = 11
    GLP_EROOT = 12
    GLP_ESTOP = 13
    GLP_EMIPGAP = 14
    GLP_ENOFEAS = 15
    GLP_ENOCVG = 16
    GLP_EINSTAB = 17
    GLP_EDATA = 18
    GLP_ERANGE = 19

    RET_CODES = {
        SUCCESS: 'LP problem instance has been successfully solved',
        GLP_EBADB:'invalid basis',
        GLP_ESING:'singular matrix',
        GLP_ECOND:'ill-conditioned matrix',
        GLP_EBOUND:'invalid bounds',
        GLP_EFAIL:'solver failed',
        GLP_EOBJLL:'objective lower limit reached',
        GLP_EOBJUL:'objective upper limit reached',
        GLP_EITLIM:'iteration limit exceeded',
        GLP_ETMLIM:'time limit exceeded',
        GLP_ENOPFS: 'no primal feasible solution',
        GLP_ENODFS: 'no dual feasible solution',
        GLP_EROOT: 'root LP optimum not provided',
        GLP_ESTOP: 'search terminated by application',
        GLP_EMIPGAP: 'relative mip gap tolerance reached',
        GLP_ENOFEAS: 'no primal/dual feasible solution',
        GLP_ENOCVG: 'no convergence',
        GLP_EINSTAB: 'numerical instability',
        GLP_EDATA: 'invalid data',
        GLP_ERANGE: 'result out of range',
    }

    # Message levels
    GLP_MSG_OFF = 0
    GLP_MSG_ERR = 1
    GLP_MSG_ON = 2
    GLP_MSG_ALL = 3
    GLP_MSG_DBG = 4

    # Simplex methods
    GLP_PRIMAL = 1
    GLP_DUALP = 2
    GLP_DUAL = 3

    # control structure
    class glp_smcp(ctypes.Structure):
        _fields_ = [
            ('msg_lev', ctypes.c_int),
            ('meth', ctypes.c_int),
            ('pricing', ctypes.c_int),
            ('r_test', ctypes.c_int),
            ('tol_bnd', ctypes.c_double),
            ('tol_dj', ctypes.c_double),
            ('tol_piv', ctypes.c_double),
            ('obj_ll', ctypes.c_double),
            ('obj_ul', ctypes.c_double),
            ('it_lim', ctypes.c_int),
            ('tm_lim', ctypes.c_int),
            ('out_frq', ctypes.c_int),
            ('out_dly', ctypes.c_int),
            ('presolve', ctypes.c_int),
            ('excl', ctypes.c_int),
            ('shift', ctypes.c_int),
            ('aorn', ctypes.c_int),
            ('foobar', ctypes.c_double*33),
        ]

    _lib.glp_init_smcp.restype = ctypes.c_int
    _lib.glp_init_smcp.argtypes = [ctypes.POINTER(glp_smcp)]

    # Simplex driver
    _lib.glp_simplex.restype = ctypes.c_int
    _lib.glp_simplex.argtypes = [ctypes.POINTER(glp_prob), ctypes.POINTER(glp_smcp)]

    # Cleanup
    _lib.glp_delete_prob.restype = None
    _lib.glp_delete_prob.argtypes = [ctypes.POINTER(glp_prob)]
