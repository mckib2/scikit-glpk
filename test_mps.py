'''Example showing how to use mpsread/write.'''

from scipy.optimize import linprog

from glpk import mpsread

if __name__ == '__main__':

    libpath = '/home/nicholas/Downloads/glpk-4.65/src/.libs/libglpk.so'

    filename = '~/Documents/HiGHS/check/instances/25fv47.mps'

    lp = mpsread(filename, libpath=libpath)
    c, A_ub, b_ub, A_eq, b_eq, bounds = lp

    if A_ub is not None:
        print(A_ub.nnz)
        A_ub = A_ub.todense()
        print(A_ub)
        print(len(b_ub), c.shape)
        print(A_ub.shape)
    if A_eq is not None:
        A_eq = A_eq.todense()

    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
    print(res)
