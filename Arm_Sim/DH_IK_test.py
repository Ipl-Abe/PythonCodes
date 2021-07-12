import numpy
from sympy import *
import sys
import os
import csv
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from time import *
import math

def createModifiedDHMatrix(alpha, a, d, q):
    mat = Matrix([[           cos(q),             -sin(q),           0,             a],
                  [sin(q)*cos(alpha),   cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                  [sin(q)*sin(alpha),   cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                  [0,0,0,1]])
    #print(type(mat))
    return mat

def createStandardDHMatrix(alpha, a, d, q):
    mat = Matrix([[      cos(q),  -sin(q)*cos(alpha),   sin(q)*sin(alpha),  cos(q)*a],
                  [      sin(q),   cos(q)*cos(alpha),  -cos(q)*sin(alpha),  sin(q)*a],
                  [           0,          sin(alpha),          cos(alpha),         d],
                  [           0,                   0,                   0,         1]])
    return mat


def main():
    home = os.path.expanduser("~")
    home = home + "/git/PythonCodes/Arm_Sim/"
    print(home)
    #path = home + 'DH_Parameter.csv'
    ur5e_StandParam_path = home + 'DH_Parameter_Standard_SimpleScala.csv'
    ur5e_ModifParam_path = home + 'DH_Parameter_Modified_SimpleScala.csv'


    f_st = open(ur5e_StandParam_path)
    f_md = open(ur5e_ModifParam_path)
    
    #f = open(path)
    reader_st = csv.reader(f_st)
    header_st = next(reader_st)

    reader_md = csv.reader(f_md)
    header_md = next(reader_md)
    
    Modified_Mat = createModifiedDHMatrix(0.0, 0.0, 0.0, 0.0)
    Standard_Mat = createStandardDHMatrix(0.0, 0.0, 0.0, 0.0)
    # l = [row for row in reader]
    # print(l)


    # mylist = []
    # for i in range(1, 4):
    #     sublist = []
    #     for j in range(1,3):
    #         sublist.append(i * 10 + j)
    #     mylist.append(sublist)

    # print(mylist)

    # print(len(reader_st.next()))
    # test_mat = [[]]
    # print(test_mat)

    # array size_of(4, 4, link_num)
    test_mat = []
    test_mat2 = []
    # for i in reader_st:
    #     test_mat.append(createModifiedDHMatrix(0.0, 0.0, 0.0, 0))
    # # print(type(array))
    # print(array[1])

    # for i in reader_st:
    #     test_mat[reader_st.__sizeof__()][len(reader_st.next())-i] = Matrix.zeros(4,4)

    for i in reader_st:
        #M = createStandardDHMatrix(float(i[0]), float(i[1]), float(i[2]), i[3])
        M = createStandardDHMatrix(float(i[0]), float(i[1]), float(i[2]), 0.0)
        Standard_Mat = Standard_Mat * M
        # print(Standard_Mat)
        test_mat.append(Standard_Mat)

    for i in reader_md:   
        M = createModifiedDHMatrix(float(i[0]), float(i[1]), float(i[2]), 0.0)
        Modified_Mat = Modified_Mat * M
        test_mat2.append(Modified_Mat)

    print("Standard : ", Standard_Mat)
    print("Modified : ", Modified_Mat) 
    print("test mat : ", test_mat)
    print("test mat2 : ", test_mat2)

    fig = plt.figure("fig1")
    ax = plt.axes(projection="3d")

    ax.clear()

    #print(len(test_mat))

    #print(test_mat[1:2])

    # for link in test_mat:
    for i in range(len(test_mat)):
        #temp = test_mat[i:i+1]
        if i == 0:
            temp1 = test_mat2[i]  
            # print(temp1)
            # print(temp1[1,1])
            ax.plot([0.0, temp1[0,3]],[0.0, temp1[1,3]],[0.0, temp1[2,3]],color='r')
        
        else:
            temp1 = test_mat2[i-1]
            temp2 = test_mat2[i]
            ax.plot([temp1[0,3], temp2[0,3]],[temp1[1,3], temp2[1,3]],[temp1[2,3], temp2[2,3]],color='b')
            
#
        # #temp = [[B_vectors[i,j],C_vectors[j]] for j in range(3)]
        # ax.plot(temp[0],temp[1],temp[2],color='y')
    
        # #temp = [[Base[j], A_vectors[i,j]] for j in range(3)]
        # ax.plot(temp[0],temp[1],temp[2],color='r')

        # #temp = [[Base[j], C_vectors[j]] for j in range(3)]
        # ax.plot(temp[0],temp[1],temp[2],color='g')

    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-1,2])  
    # # plot all links and joints
    # fig = visualize(A,B,C,x,y,z)
    plt.show(fig)

def visualize(A,B,C,x,y,z):

    fig = plt.figure("fig1")
    ax = plt.axes(projection="3d")

    thetas = solves(A,B,C,x,y,z)
    print("thetas")
    print(np.rad2deg(thetas))
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

    print(my_norm(C_vectors - A_vectors[0]))

    Theta1 = cos_formula(B,C,my_norm(C_vectors - A_vectors[0]))
    print("angle test")
    print(np.rad2deg(Theta1))

    Theta2 = cos_formula(B,C,my_norm(C_vectors - A_vectors[1]))
    print("angle test2")
    print(np.rad2deg(Theta2))

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

    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-1,2])    
    return fig


if __name__ == "__main__":
    main()

    # x = Symbol('x')
    # y = Symbol('y') 
    # #expr=x**2+y**2
    # q1 = Symbol('q1')
    # res = sin(q1)
    # print(res) 

