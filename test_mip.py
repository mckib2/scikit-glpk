
from glpk import glpk

c = [8, 1]
A_ub = [
    [-1, -2],
    [-4, -1],
    [2, 1],
]
b_ub = [14, -33, 20]

res = glpk(c, A_ub, b_ub, solver='mip', mip_options={
    'nomip': False,
    'intcon': [1],
    'presolve': True,
})
print(res)
