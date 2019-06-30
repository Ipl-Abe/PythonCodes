import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib as mpl
from sympy import *
from sympy.abc import *
import matplotlib.animation as animation
from time import *

def radtoDeg(x):
    return x * np.pi / 180

def degtoRad(x):
    return x * 180 / np.pi

def norm2(x):
    return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]

def my_norm(x):
    s = norm2(x)
    return np.sqrt(s)



def symbol_calc():
    basePhi = symbols("phi")
    thetas = symbols("theta_0 theta_1")
    angles = [0,pi]
    unit_vectors = [Matrix([[cos(angles[i])], [sin(angles[i])]]) for i in range(2)]
    params = [(S,1.0),(r,1.0),(l,1.0)]
    params.append((angles[0],0))
    params.append((angles[1],pi))

    params.append((x,0))    
    params.append((y,0))
    ey = Matrix([0,0,1])


    #print(unit_vectors)

    S_vectors = np.array([
        [r*unit_vectors[0]],
        [r*unit_vectors[1]]
        ])
    # print("svec")
    # print(S_vectors[0].T)
    RMat = Matrix([
        [cos(basePhi), -sin(basePhi)],
        [sin(basePhi),  cos(basePhi)]
        ])

    P_vector = Matrix([x, y])
    #print(P_vector)
    L_vector =  np.array([
        P_vector + RMat * S_vectors[0].T ,
        P_vector + RMat * S_vectors[1].T
        ])   
    # print("L_vector")
    # print(L_vector[0])

    a_vector = np.array([1,0])
    a_vector = a_vector[:,np.newaxis]
   # print(L_vector)

    c1 = L_vector * a_vector.T
    c2 = L_vector[0,0] * L_vector[0,0] + L_vector[0,1] * L_vector[0,1]
    c3 = c1[0,0]**2 - c2**2 + l**2
    print(c3)
    #print(c1[1,0])

    C_vector = c1[0,0] + sqrt(c3)

    print(C_vector)


def solves(r,l,x,y,phi):
    #Calc Angles
    abs = np.abs
    cos = np.cos   
    sin = np.sin 

    pi = np.pi
    atan = np.arctan
    sqrt = np.sqrt
    angles = [0,pi]
    unit_vectors = np.array([np.cos(angles), np.sin(angles)]).T
    #print(unit_vectors)

    S_vectors = np.array([
        [r*unit_vectors[0]],
        [r*unit_vectors[1]]
        ])
    print("svec")
    print(S_vectors[0].T)
    RMat = Matrix([
        [cos(phi), -sin(phi)],
        [sin(phi),  cos(phi)]
        ])

    P_vector = np.array([[x], [y]])
    print(P_vector)
    L_vector =  np.array([
        P_vector + RMat * S_vectors[0].T ,
        P_vector + RMat * S_vectors[1].T
         ])   
    print("L_vector")
    print(L_vector)

    a_vector = np.array([1,0])
    a_vector = a_vector[:,np.newaxis]

    print("a_vector")
    print(a_vector)

    c1 = L_vector[0] * a_vector.T
    c2 = L_vector[0,0] * L_vector[0,0] + L_vector[0,1] * L_vector[0,1]
    c3 = c1[0,0]**2 - c2 + l**2
    print("c1")
    print(c1)
    print(c2)
    print(c3)

    
    C_vector = c1[0,0] + np.sqrt(0.75)
    ass = 0.5/C_vector
    print(ass)
    ccc = atan(0.129333)
    print("C_vector")
    print(C_vector)
    print("atan", ccc)



if __name__ == "__main__":
    #symbol_calc() 
    r = 1.0
    l = 1.0
    x = 2.0
    y = 0.5
    phi = np.deg2rad(0.0)
    solves(r,l,x,y,phi)
