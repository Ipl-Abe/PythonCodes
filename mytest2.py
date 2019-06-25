import os
import numpy as np
from matplotlib import pyplot as plt

mat1 = np.array([1,2])
mat3 = np.array([[1,2],[3,4]])
mat1.ndim
print(mat1)
print(mat3)
b = np.reshape(mat3,(1,4))
c = np.reshape(mat3,(4,1))
print(mat3.T)
print(b)
print(c)

mat2 = mat1[:,np.newaxis]
mat2[0,0] = 3
mat2[1,0] = 4
#mat2[2,2] = 4
print(mat2)

plt.plot(mat1[0],mat1[1],c=None,marker='o')

plt.show()
