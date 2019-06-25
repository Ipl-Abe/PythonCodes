
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sympy import *
from sympy.abc import *

def norm2(x):
    return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]

angles = symbols("phi_0 phi_1 phi_2")
thetas = symbols("theta_0 theta_1 theta_2")
unit_vectors = [Matrix([cos(angles[i]),sin(angles[i]),0]) for i in range(3)]
ez = Matrix([0,0,1])


params = [(A,130.0),(B,200.0),(C,400.0),(D,130.0)]
params.append((angles[0],2.0*np.pi/3.0*0))
params.append((angles[1],2.0*np.pi/3.0*1))
params.append((angles[2],2.0*np.pi/3.0*2))
params.append((x,0))
params.append((y,0))
params.append((z,400))

A_vectors = map(lambda x: A*x,unit_vectors)
B_vectors = [A_vectors[i] + B*(unit_vectors[i] * cos(thetas[i])-ez*sin(thetas[i]) ) for i in range(3)]
D_vector = Matrix([x,y,z])
C_vectors = [D_vector + D * unit_vectors[i] for i in range(3)]

print(A_vectors)

eq0 = C**2-norm2((C_vectors[0] - B_vectors[0]))
print(eq0)
eq0 = expand(eq0)
print("expand")
print(eq0)
eq0 = simplify(eq0)