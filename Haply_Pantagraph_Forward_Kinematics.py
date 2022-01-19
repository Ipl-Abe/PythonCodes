import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib as mpl
from sympy import *
from sympy.abc import *
import matplotlib.animation as animation
from time import *
import scipy.linalg as linalg
import math

l = 0.07
L = 0.09
d = 0.00

def radtoDeg(x):
    return x * np.pi / 180

def degtoRad(x):
    return x * 180 / np.pi

def norm2(x):
    return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]

def my_norm(x):
    s = norm2(x)
    return np.sqrt(s)


def solves(A,B,C,x,y,z):
    #Calc Angles
    abs = np.abs
    cos = np.cos   
    sin = np.sin 

    pi = np.pi
    atan = np.arctan
    sqrt = np.sqrt

    #Angle
    thetas = np.zeros((2,))
    angles = np.array([pi * (2.0*(i)/2.0) * i for i in range(2)])
    for i in range(2):
        phi_0 = angles[i]
        P = -A**2 + 2*A*x*cos(phi_0) + 2*A*y*sin(phi_0) + C**2 - x**2 - y**2 - z**2
        Q = 2*B*z
        R = -2*A*B + 2*B*x*cos(phi_0) + 2*B*y*sin(phi_0)
        print ("P ",P," Q ", Q, " R ", R)
        theta_0 = -2*atan((Q - sqrt(-P**2 + Q**2 + R**2))/(P-R))
        theta_1 = -2*atan((Q + sqrt(-P**2 + Q**2 + R**2))/(P-R))

        thetas[i] = theta_1 if abs(theta_0) > abs(theta_1) else theta_0
    return thetas

def symbol_calc(angles):

    l1 = l
    l2 = l
    L1 = L
    L2 = L
    #angles = np.array([0,150])
    
    th1 = math.pi/180*angles[0]
    th2 = math.pi/180*angles[1]
    #th1 = angles[0]
    #th2 = angles[1]

    # Forward Kinematics
    c1 = cos(th1)
    c2 = cos(th2)
    s1 = sin(th1)
    s2 = sin(th2)
    xA = l1*c1
    yA = l1*s1
    xB = d+l2*c2   
    yB = l2*s2
    hx = xB-xA
    hy = yB-yA 
    hh = math.pow(hx,2) + math.pow(hy,2)
    hm = sqrt(hh)
    cB = - (math.pow(L2,2) - math.pow(L1,2) - hh) / (2*L1*hm) 
    
    h1x = L1*cB * hx/hm 
    h1y = L1*cB * hy/hm 
    h1h1 = math.pow(h1x,2) + math.pow(h1y,2) 
    h1m = math.sqrt(h1h1) 
    sB = math.sqrt(1-math.pow(cB,2)) 
     
    lx = -L1*sB*h1y/h1m 
    ly = L1*sB*h1x/h1m 
    
    x_P = xA + h1x + lx
    y_P = yA + h1y + ly 
     
    phi1 = math.acos((x_P-l1*c1)/L1)
    phi2 = math.acos((x_P-d-l2*c2)/L2)
     
    c11 = math.cos(phi1) 
    s11 = math.sin(phi1) 
    c22 = math.cos(phi2) 
    s22 = math.sin(phi2) 
  
    dn = L1 *(c11 * s22 - c22 * s11)
    eta = (-L1 * c11 * s22 + L1 * c22 * s11 - c1 * l1 * s22 + c22 * l1 * s1)  / dn
    nu = l2 * (c2 * s22 - c22 * s2)/dn
    
    JT11 = -L1 * eta * s11 - L1 * s11 - l1 * s1
    JT12 = L1 * c11 * eta + L1 * c11 + c1 * l1
    JT21 = -L1 * s11 * nu
    JT22 = L1 * c11 * nu

    x_E = x_P
    y_E = y_P    

    print(x_P)
    print(y_P)



    ## another forfard kinematics method
    c1=cos(th1)
    c2=cos(th2)
    s1=sin(th1)
    s2=sin(th2)
    xA=l*c1
    yA=l*s1
    xB=d+l*c2
    yB=l*s2
    R=pow(xA,2)+pow(yA,2)
    S=pow(xB,2)+pow(yB,2)
    M=(yA-yB)/(xB-xA)
    N=0.5*(S-R)/(xB-xA)
    a=pow(M,2)+1
    b=2*(M*N-M*xA-yA)
    c=pow(N,2)-2*N*xA+R-pow(L,2)
    Delta=pow(b,2)-4*a*c

    # Result of forward kinematics - X and Y of the endpoint
    y_h=(-b+sqrt(Delta))/(2*a)
    x_h=M*y_h+N

    # Jacobian values
    phi1=acos((x_h-l*c1)/L)
    phi2=acos((x_h-d-l*c2)/L)
    s21=sin(phi2-phi1)
    s12=sin(th1-phi2)
    s22=sin(th2-phi2)
    J11=-(s1*s21+sin(phi1)*s12)/s21
    J12=(c1*s21+cos(phi1)*s12)/s21
    J21=sin(phi1)*s22/s21
    J22=-cos(phi1)*s22/s21

    print(x_h)
    print(y_h)


#     plt.plot(x_P,y_P,c=None,marker='o')

#     plt.plot(xA,yA,c=None,marker='o')
#     plt.plot(xB,yB,c=None,marker='o')
#     plt.plot(h1x,h1y,c=None,marker='o')

#     plt.plot([0,xA],[0,yA])
#     plt.plot([0,xB],[0,yB])
#  #   plt.plot([x_P,xB],[y_P,yB])

#     XXX = xA + L1*c11
#     YYY = yA + L1*s11
#     plt.plot([xA,XXX],[yA,YYY])

#     plt.xlim([-0.1, 0.1])
#     plt.ylim([-0.1, 0.1])
#     plt.show()

def cos_formula(B,C,A):
    cosineA = (B**2 + C**2 - A**2) / (2*B*C)
    theta = np.arccos(cosineA)
    return theta

def visualize(A,B,C,x,y,z):

    fig = plt.figure("fig1")
    ax = plt.axes(projection="3d")

    thetas = solves(A,B,C,x,y,z)

    print("thetas")
    print(np.rad2deg(thetas))

    symbol_calc(thetas)  

    angles = np.array([np.pi * (2.0*(i)/2.0)*i for i in range(2)])
    unit_vectors = np.array([np.cos(angles), np.sin(angles), np.zeros(2)]).T
    ez = np.array([0,0,1])

    A_vectors = A * unit_vectors
    print("A_vectors")
    print(A_vectors)
    B_vectors = np.array([A_vectors[i] + 
                          B*(unit_vectors[i] * np.cos(thetas[i]) + ez* np.sin(thetas[i]) )for i in range(2)])

    print("B_vectors")
    print(B_vectors)


    C_vectors = np.array([x,y,z])   

    # Theta1 = cos_formula(B,C,my_norm(C_vectors - A_vectors[0]))
    # print("angle test")
    # print(np.rad2deg(Theta1))

    # Theta2 = cos_formula(B,C,my_norm(C_vectors - A_vectors[1]))
    # print("angle test2")
    # print(np.rad2deg(Theta2))


    # if math.isnan(Theta1) == True and math.isnan(Theta2) == True:
    #     return False

    index = np.arange(3) % 2
    #print(A_vectors)
    #print(C_vectors[2])
    ax.clear()
    #ax.plot(A_vectors,color='r')
    #ax.plot(xs = B_vectors[0],ys = B_vectors[1],zs = B_vectors[2],color='g')
    #ax.plot(xs = C_vectors[0],ys = C_vectors[1],zs = C_vectors[2],color='g')
    #ax.plot(xs = A_vectors[0,0],ys=A_vectors[0,1],zs=A_vectors[0,2],color='r')
    #ax.plot(xs = C_vectors[0,0],ys=C_vectors[0,1],zs=C_vectors[0,2],color='g')
    
    Base = np.array([0.0,0.0,0.0])  
    for i in range(2):
        temp = [[A_vectors[i,j], B_vectors[i,j]] for j in range(3)]
        #print(temp)
        ax.plot(temp[0],temp[1],temp[2],color='b')
#
        temp = [[B_vectors[i,j],C_vectors[j]] for j in range(3)]
        ax.plot(temp[0],temp[1],temp[2],color='y')
    
        temp = [[Base[j], A_vectors[i,j]] for j in range(3)]
        ax.plot(temp[0],temp[1],temp[2],color='r')

        temp = [[Base[j], C_vectors[j]] for j in range(3)]
        ax.plot(temp[0],temp[1],temp[2],color='g')

    ax.set_xlim([-0.1,0.1])
    ax.set_ylim([-0.1,0.1])
    ax.set_zlim([-0.1,0.1])  


    plt.show()
    return fig

if __name__ == "__main__":
    #symbol_calc() 
    d = 0.0
    l = 0.07
    L = 0.09
    x = 0.03
    y = 0.0
    z = -0.15

    print("x : ", x)
    print("y : ", z)

    phi = np.deg2rad(0)
    
    find = solves(d,l,L,x,y,z)

    #fig = plt.figure("fig1")

    # if find

    fig = visualize(d,l,L,x,y,z)
    #solves(r,l,x,y,phi)
    # fig = visualize(r,l,x,y,phi)

    # plt.show(fig)

