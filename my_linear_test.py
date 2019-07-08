import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib as mpl
from sympy import *
from sympy.abc import *
import matplotlib.animation as animation
from time import *
import numpy.linalg as LA

def radtoDeg(x):
    return x * np.pi / 180

def degtoRad(x):
    return x * 180 / np.pi

def norm2(x):
    return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]

def my_norm(x):
    s = norm2(x)
    return np.sqrt(s)


def solve_Jacobi(c,l,base_theta,phi):
    
    x1 = c[0] + l*cos(phi[0])
    x2 = c[1] + l*cos(phi[1])

    y1 = l * sin(phi[0]) 
    y2 = l * sin(phi[1]) 
    
    #x_vector = 1/2 *  ( x1 + x2 )
    #y_vector = 1/2 *  ( y1 + y2 )     
    #tan = (y2 - y1)/(x2 - x1) 

    pp = (-l*(l*sin(phi[1]) - l*sin(phi[0]))*sin(phi[0])/(-c[0] + c[1] + l*cos(phi[1]) - l*cos(phi[0]))**2 - l*cos(phi[0])/(-c[0] + c[1] + l*cos(phi[1]) - l*cos(phi[0])))/((l*sin(phi[1]) - l*sin(phi[0]))**2/(-c[0] + c[1] + l*cos(phi[1]) - l*cos(phi[0]))**2 + 1)
    print(pp)

    pp2 = (l*(l*sin(phi[1]) - l*sin(phi[0]))*sin(phi[1])/(-c[0] + c[1] + l*cos(phi[1]) - l*cos(phi[0]))**2 + l*cos(phi[1])/(-c[0] + c[1] + l*cos(phi[1]) - l*cos(phi[0])))/((l*sin(phi[1]) - l*sin(phi[0]))**2/(-c[0] + c[1] + l*cos(phi[1]) - l*cos(phi[0]))**2 + 1)
    print(pp2)

    Jacobi = np.array([
        [-0.5*l*sin(phi[0]), 0.5*l*cos(phi[0]), 
        pp[0]],
        
        [-0.5*l*sin(phi[1]), 0.5*l*cos(phi[1]),
        pp2[0]]
    ])
    print(Jacobi)
    return Jacobi


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
    print("c3 : ")
    print(c3)
    #print(c1[1,0])

    C_vector = c1[0,0] + sqrt(c3)
    print("c_vector : ")
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
    # OK
    unit_vectors = np.array([np.cos(angles), np.sin(angles)]).T
    #print(unit_vectors)

    S_vectors = np.array([[r*unit_vectors[0]],[r*unit_vectors[1]]])
    #print("svec")
    #print(S_vectors[0].T)
    # RMat = Matrix([
    #     [cos(phi), -sin(phi)],
    #     [sin(phi),  cos(phi)]
    #     ])
    RMat = np.array([
        [cos(phi), -sin(phi)],
        [sin(phi),  cos(phi)]
    ])

    P_vector = np.array([[x], [y]])
    # print("RMat")
    # print(RMat)
       # B_vectors = np.array([A_vectors[i] + 
        #                  B*(unit_vectors[i] * np.cos(thetas[i]) + ez* np.sin(thetas[i]) )for i in range(2)])
    L_vector1 =  np.array(S_vectors[0]*RMat) + P_vector
    L_vector2 =  np.array(S_vectors[1]*RMat) + P_vector
    L_vector = np.array([L_vector1[:,0],L_vector2[:,0]])
    print("L_vector")
    print(L_vector)

    a_vector = np.array([1,0])
    a_vector = a_vector[:,np.newaxis]

    # print("L_vector")
    # print(L_vector[0,0] * L_vector[0,0] + L_vector[0,1] * L_vector[0,1])

    c1 = L_vector * a_vector.T
    c2 = [L_vector[0,0] * L_vector[0,0] + L_vector[0,1] * L_vector[0,1],
          L_vector[1,0] * L_vector[1,0] + L_vector[1,1] * L_vector[1,1]]
    c3 = [c1[0,0]**2 - c2[0] + l**2,   c1[1,0]**2 - c2[1] + l**2 ]
    print("c1")
    print(c1)
    print("c2")
    print(c2)
    print("c3")
    print(c3)

    
    
    C_vector = np.array([
        [c1[0,0] + np.sqrt(c3[0]), c1[0,1]],
        [c1[1,0] - np.sqrt(c3[1]), c1[1,1]]
        ])
    #ass = 0.5/C_vector
    # print(ass)
    #ccc = atan(0.129333)
    print("C_vector")
    print(C_vector)
    # print("atan", ccc)

    return C_vector


def visualize(r,l,x,y,phi):
    #Calc Angles
    abs = np.abs
    cos = np.cos   
    sin = np.sin 

    pi = np.pi
    atan = np.arctan
    sqrt = np.sqrt
    angles = [0,pi]
    unit_vectors = np.array([np.cos(angles), np.sin(angles)]).T
    S_vectors = np.array([[r*unit_vectors[0]],[r*unit_vectors[1]]])
    RMat = np.array([
        [cos(phi), -sin(phi)],
        [sin(phi),  cos(phi)]
    ])
    P_vector = np.array([[x], [y]])
    L_vector1 =  np.array(S_vectors[0]*RMat) + P_vector
    L_vector2 =  np.array(S_vectors[1]*RMat) + P_vector
    L_vector = np.array([L_vector1[:,0],L_vector2[:,0]])


    fig = plt.figure()
    c1 = solves(r,l,x,y,phi)
    #print(c1)

    foot_theta = [degtoRad(60),degtoRad(60)]

    Jaco = solve_Jacobi(c1,l,phi,foot_theta)

    omega = np.array([0.1,0.1])

    omega = omega[:, np.newaxis]

    print("jaco")
    print(Jaco.shape)
    print(omega.shape)

    
    calc = np.dot(Jaco.T,omega)

    www = np.dot(Jaco,Jaco.T)

    #det = np.pinv(www)
    #det = LA.det(www)
    #det = np.linalg.det(www)

    #pausedu_Jaco = np.linalg.pinv(www)
    
    #pausedu_Jaco = np.linalg.solve(Jaco, Jaco.T)

    # print("calc")
    # print(calc)


    print("manipulability")
    print(det)

   # print("pausedu")
   # print(pausedu_Jaco)

    # visualize
    plt.plot(0,0,"o")
    temp1 = np.array([[c1[0,0],L_vector[0,0]],[0,L_vector[0,1]]])
    temp2 = np.array([[c1[1,0],L_vector[1,0]],[0,L_vector[1,1]]])
   # print(L_vector)
    
    plt.plot([x,L_vector[0,0]],[y,L_vector[0,1]],"-o",color='b')
    plt.plot([x,L_vector[1,0]],[y,L_vector[1,1]],"-o",color='b')
    plt.plot(temp1[0],temp1[1],"-o",color='r')
    plt.plot(temp2[0],temp2[1],"-o",color='r')
    plt.plot(0,0,color="k")


    plt.xlim([-2,2.0])
    plt.ylim([-2.0,2.0])
    return fig

if __name__ == "__main__":
    #symbol_calc() 
    r = 0.5
    l = 0.5
    x = 0.0
    y = 0.5
    phi = np.deg2rad(0)
    #solves(r,l,x,y,phi)
    fig = visualize(r,l,x,y,phi)

    plt.show(fig)



