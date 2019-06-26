import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib as mpl
from sympy import *
from sympy.abc import *
import matplotlib.animation as animation
from time import *

def radtoDeg(x):
    return x * pi / 180

def degtoRad(x):
    return x * 180 / pi

def norm2(x):
    return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]

def symbol_calc():
    angles = symbols("phi_0 phi_1 ")
    thetas = symbols("theta_0 theta_1")

    #unit_vectors = np.array([[1,0],[-1,0]])
    unit_vectors = [Matrix([cos(angles[i]), sin(angles[i]),0]) for i in range(2)]
    #unit_vectors2 = np.array([[1,0],[-1,0]])
    
    #print(unit_vectors)
    params = [(A,1.0),(B,2.0)]
    params.append((angles[0],0))
    params.append((angles[1],pi))

    params.append((x,0))    
    params.append((y,1))
    params.append((z,1))
    ey = Matrix([0,0,1])
    
    #A_vectors = A*unit_vectors[0]
    #A_vectors = map(lambda x: A*x, unit_vectors)
    A_vectors = [A*unit_vectors[0], A*unit_vectors[1]]
    #A_vectors2 = [A*unit_vectors2[0], A*unit_vectors2[1]]
    #unit_vectors
    #print(unit_vectors[0])
    #print(A_vectors)
    #print(A_vectors2)
    #print("Debug")

    B_vectors = [A_vectors[i] + B*(unit_vectors[i]*cos(thetas[i])+ey*sin(thetas[i])) for i in range(2)]
    #print("BVector")
    #print(B_vectors)
  
   
    C_vectors = Matrix([x,y,z])
    #print(C_vectors)
    #print(A_vectors -B_vectors)
    #UU = C_vectors - B
    #print("test : ")
    #print(norm2((C_vectors-B_vectors[0])))
    eq0 = C**2 - norm2((C_vectors-B_vectors[0]))
    

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

    #print("Q_dash test : ")
    #print(expand(Q_dash * sin(thetas[0])))
    


    return 1


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
    #angles = np.array([pi * (2.0/3.0) * i for i in range(2)])
    angles = np.array([pi * (2.0*(i)/2.0) * i for i in range(2)])
    #print("angles : ")
    #print(angles)
    for i in range(2):
        phi_0 = angles[i]

        #P = -A**2 + 2*A*x*cos(phi_0) + 2*A*y*sin(phi_0) - B**2 + C**2 - x**2 - y**2 - z**2
        P = -A**2 + 2*A*x*cos(phi_0) + 2*A*y*sin(phi_0) + C**2 - x**2 - y**2 - z**2
        Q = 2*B*z
        R = -2*A*B + 2*B*x*cos(phi_0) + 2*B*y*sin(phi_0)

        #P = -A**2 + 2*A*D + 2*A*x*cos(phi_0) + 2*A*y*sin(phi_0)-B**2 + C**2 -D**2 -2*D*x*cos(phi_0) -2*D*y*sin(phi_0)-x**2 - y**2 - z**2
        
        #Q = -2*B*z
        #R = -2*A*B + 2*B*D + 2*B*x*cos(phi_0) + 2*B*y*sin(phi_0)

        theta_0 = -2*atan((Q - sqrt(-P**2 + Q**2 + R**2))/(P-R))
        theta_1 = -2*atan((Q + sqrt(-P**2 + Q**2 + R**2))/(P-R))

        thetas[i] = theta_1 if abs(theta_0) > abs(theta_1) else theta_0
    return thetas

def visualize(A,B,C,x,y,z):

    fig = plt.figure("fig1")
    ax = plt.axes(projection="3d")

    thetas = solves(A,B,C,x,y,z)
    #print("thetas")
    #print(thetas)
    angles = np.array([np.pi * (2.0*(i)/2.0)*i for i in range(2)])
    unit_vectors = np.array([np.cos(angles), np.sin(angles), np.zeros(2)]).T
    ez = np.array([0,0,1])

    A_vectors = A * unit_vectors
    #print("A_vectors")
    #print(A_vectors)
    B_vectors = np.array([A_vectors[i] + 
                          B*(unit_vectors[i] * np.cos(thetas[i]) + ez* np.sin(thetas[i]) )for i in range(2)])

    #print("B_vectors")
    #print(B_vectors)


    C_vectors = np.array([x,y,z])   

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

    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    ax.set_zlim([-5,1])    
    return fig



if __name__ == "__main__":
    #symbol_calc()
    A = 0.5
    B = 1.0
    C = 0.6

    x = 0.5
    y = 0.0
    z = 1.3

    fig = plt.figure("fig1")
    #ax = plt.axes(projection="3d")
    #ax = fig.add_subplot(111, projection='3d')
    #ims = []
    
    fig = visualize(A,B,C,x,y,z)

    #for i in range(10):

        #r = 0.3
     #   y = 0.0
     #   x = r*np.cos(0.4*i)
     #   z = r*np.sin(0.4*i) + 0.5
     #   ims.append(visualize(A,B,C,x,y,z))
        #fig = visualize(A,B,C,x,y,z,fig)
        #plt.show()
        #plt.show(fig)
        #plt.show()
        #sleep(1)
       
        #ims.append(fig)
        
        #plt.
    #print(ims)
    #ani = animation
    #ani.ArtistAnimation(fig, ims, interval = 100,blit=True)
    #ani = animation.ArtistAnimation(fig, ims, interval = 100,blit=True)
    #ani = animation.FuncAnimation(fig,ims,19,interval=40)
    plt.show(fig)
    #ims.clear()
    print("Fin")




