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
    P_vectors = Matrix([x,y])
    base_Rot = symbols("phi_0")
    thetas = symbols("theta_0 theta_1")
    angles = [0,pi]
    unit_vectors = [Matrix([cos(angles[i]), sin(angles[i])]) for i in range(2)]
    params = [(S,0.1),(D,0.8)]
    params.append((angles[0],0))
    params.append((angles[1],pi))

    params.append((x,0))    
    params.append((y,0))
    ey = Matrix([0,1])
    #print(unit_vectors)

    S_vectors = [S*unit_vectors[0], S*unit_vectors[1]]

    # print("S_vector")
    # print(S_vectors[0])
    
    RMat = Matrix([
        [cos(base_Rot),-sin(base_Rot)],
        [sin(base_Rot),cos(base_Rot)]           
                   ])
    # print("RMat")
    # print (RMat)
    L_vector = [P_vectors + RMat * S_vectors[0], P_vectors + RMat * S_vectors[1]]

    L_1 = np.array(L_vector[0])

    print("L_1")
    print(L_1.T)

    world_vector = np.array([[1],[0]])

    print("world_vec")
    print(world_vector)

    print("L_1.T * a_vec")
    print(L_1.T * world_vector.T)

    # Calculate L^T・a
    formula1 = L_1[0] * world_vector[0] + L_1[1]* world_vector[1]
    formula2 = L_1[0] * L_1[0] + L_1[1] * L_1[1]
     
    c_vec_root1 = [formula1[0] + sqrt((formula1[0])**2 + D**2 +(formula2[0])**2)]
    
    print("c_vec_rooot")
    print(formula1)
    print(c_vec_root1)


# R : Rotation Angle for Base
def solves(r,L,x,y,z,R):
    #Calc Angles
    abs = np.abs
    cos = np.cos   
    sin = np.sin 

    pi = np.pi
    atan = np.arctan
    sqrt = np.sqrt
    
    P_vectors = np.array([x,y])
    #Angle
    thetas = np.zeros((2,))
    angles = np.array([pi * (2.0*(i)/2.0) * i for i in range(2)])
    angles = [0,pi]
    unit_vectors = np.array([([cos(angles[i]), sin(angles[i])]) for i in range(2)])
    S_vectors = [r*unit_vectors[0], r*unit_vectors[1]]

    print(unit_vectors)
    print(S_vectors)

    # Debug
    print("Debug")

    for i in range(2):
        phi_0 = angles[i]

        RMat = np.array([[np.cos(R), -np.sin(R)],[np.sin(R), np.cos(R)]])

        L_vector = [P_vectors + RMat * S_vectors[0], P_vectors + RMat * S_vectors[1]]        

    return thetas




if __name__ == "__main__":
    #symbol_calc()
    r = 1.0
    L = 1.0
    x = 3.0
    y = 0.8
    R = degtoRad(0)
    solves(r,L,x,y,z,R)

