import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib as mpl
from sympy import *
from sympy.abc import *
import matplotlib.animation as animation
from time import *
import math

def radtoDeg(x):
    return x * math.pi / 180

def degtoRad(x):
    return x * 180 / math.pi

def norm2(x):
    return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]

def my_norm(x):
    s = norm2(x)
    return np.sqrt(s)



def symbol_calc():
    #angles = symbols("phi_0 phi_1 ")
    thetas = symbols("theta_0 theta_1")
    angles = [0,pi]
    unit_vectors = [Matrix([cos(angles[i]), sin(angles[i]),0]) for i in range(2)]
    params = [(A,0.5),(B,0.8)]
    params.append((angles[0],0))
    params.append((angles[1],pi))

    params.append((x,0))    
    params.append((y,0))
    params.append((z,1.0))
    ey = Matrix([0,0,1])
    print(unit_vectors)


    A_vectors = [A*unit_vectors[0], A*unit_vectors[1]]
    print("A_vector")
    print(A_vectors)
    
    B_vectors = [A_vectors[i] + B*(unit_vectors[i]*cos(thetas[i])+ey*sin(thetas[i])) for i in range(2)]
    print("BVector")
    print(B_vectors)
    

    C_vectors = Matrix([x,y,z])
    eq0 = C**2 - norm2((C_vectors - B_vectors[0]))
    print("DDebug")
    print(eq0)

    eq0 = expand(eq0)
    print(eq0)
    eq0 = simplify(eq0)
    print("Simplify")
    print(eq0)


        # Next Step
    formula = Eq(P +Q * sin(thetas[0]) + R * cos(thetas[0]),0)
    #print(formula)


    thetas_ans = solve(formula,thetas[0])
    #print(thetas_ans)

    Q_dash = eq0.coeff(sin(thetas[0]),1)
    R_dash = eq0.coeff(cos(thetas[0]),1)
    P_dash = eq0.coeff(cos(thetas[0]),0)    - Q_dash * sin(thetas[0])
    #P_dash = P_dash + B**2*sin(thetas[0])**2 -2*B*y*sin(thetas[0]) + 2*A*B*sin(angles[0])*sin(thetas[0])
    print("Q_dash : ")
    print(Q_dash)
    print("R_dash : ")
    print(R_dash)
    print("P_dash : ")
    print(P_dash)

if __name__ == "__main__":
    symbol_calc()


