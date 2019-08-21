import scipy.linalg as linalg
import numpy as np

A = np.array([[6, 4, 1],
              [1, 8, -2],
              [3, 2, 0]])
b = np.array([7, 6, 8])

LU = linalg.lu_factor(A)
x = linalg.lu_solve(LU, b)

print(x)