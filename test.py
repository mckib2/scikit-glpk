'''Basic example showing usage of scikit-glpk.'''

import numpy as np
from scipy.optimize import linprog

from glpk import glpk, GLPK

if __name__ == '__main__':

    c = [-1, 8, 4, -6]
    A_ub = [[-7, -7, 6, 9],
            [1, -1, -3, 0],
            [10, -10, -7, 7],
            [6, -1, 3, 4]]
    b_ub = [-3, 6, -6, 6]
    A_eq = [[-10, 1, 1, -8]]
    b_eq = [-4]
    bnds = None
    res = glpk(
        c, A_ub, b_ub, A_eq, b_eq, bnds,
        message_level=GLPK.GLP_MSG_OFF,
        maxit=100,
        timeout=10,
        solver='interior',
        simplex_options={
            'init_basis': 'adv',
            'method': 'dual',
            'presolve': True,
            'exact': True,
        })
    print('GLPK:')
    print(res)

    print('\n\n')
    print('linprog:')
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bnds)
    print(res)
