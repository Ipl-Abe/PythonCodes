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
    Phies = symbols("phi_1 phi_2")
    thetas = symbols("theta_0 theta_1")
    angles = [0,pi]
    params = [(A,1.0),(B,1.0),(C,1.0),(D,1.0),(E,1.0),(F,1.0)]
    params.append((angles[0],0))
    params.append((angles[1],pi))




    var('a b c x')


    f = a*x**2 + b*x + c


    dfdx1 = diff(f, x)


    dfdx2 = diff(f, x, 2)


    dfdx3 = diff(f, x, 3)

    print("dfdx1 = {}".format(dfdx1))
    print("dfdx2 = {}".format(dfdx2))
    print("dfdx3 = {}".format(dfdx3))
    p0 = np.array([
        [A]*x,
        [B]*x,
        [C]*x,
        [D]*x,
        [E]*x,
        [F]*x,    
    ])

    Derivative(p0)

    print(p0)



if __name__ == "__main__":
    symbol_calc() 