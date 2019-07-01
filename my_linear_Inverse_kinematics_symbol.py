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
    params = [(S,1.0),(r,1.0),(L,1.0),(I,1.0),(C,1.0),(D,1.0)]
    params.append((angles[0],0))
    params.append((angles[1],pi))

    unit_vectors = [Matrix([[cos(angles[i])], [sin(angles[i])]]) for i in range(2)]

    x1 = C + L*cos(Phies[0])
    x2 = D + I*cos(Phies[1])

    y1 = L * sin(Phies[0]) 
    y2 = I * sin(Phies[1]) 
    


    x_vector = 1/2 *  ( x1 + x2 )
    y_vector = 1/2 *  ( y1 + y2 ) 
    tan = (y2 - y1)/(x2 - x1) 
    # print("tan")
    # print(tan)
    theta_vector = atan(tan)

    F = Matrix([
        [x_vector],
        [y_vector],
        [theta_vector]
    ])

    print(F)
    # print(x_vector)
    # print(y_vector)
    # print(theta_vector)


    AAA =  array.derive_by_array(F, Phies)

    print("derive")
    print(AAA)


    #[
    #    [
    #    [-0.5*L*sin(phi_1)], 
    #    [0.5*L*cos(phi_1)],
    #    [(-L*(I*sin(phi_2) - L*sin(phi_1))*sin(phi_1)/(-C + D + I*cos(phi_2) - L*cos(phi_1))**2 - L*cos(phi_1)/(-C + D + I*cos(phi_2) - L*cos(phi_1)))/((I*sin(phi_2) - L*sin(phi_1))**2/(-C + D + I*cos(phi_2) - L*cos(phi_1))**2 + 1)]
    #    ],
    #    [
    #    [-0.5*I*sin(phi_2)],
    #    [0.5*I*cos(phi_2)], 
    #    [(I*(I*sin(phi_2) - L*sin(phi_1))*sin(phi_2)/(-C + D + I*cos(phi_2) - L*cos(phi_1))**2 + I*cos(phi_2)/(-C + D + I*cos(phi_2) - L*cos(phi_1)))/((I*sin(phi_2) - L*sin(phi_1))**2/(-C + D + I*cos(phi_2) - L*cos(phi_1))**2 + 1)]
    #    ]
    #]
    

    

if __name__ == "__main__":
    symbol_calc() 
    # r = 0.5
    # l = 0.5
    # x = 0.0
    # y = 0.2
    # phi = np.deg2rad(10)
    # #solves(r,l,x,y,phi)
    # fig = visualize(r,l,x,y,phi)

    # plt.show(fig)