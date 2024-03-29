import numpy
from sympy import *
import sys
import os
import csv


def createModifiedDHMatrix(alpha, a, d, q):
    mat = Matrix([[           cos(q),             -sin(q),           0,             a],
                  [sin(q)*cos(alpha),   cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                  [sin(q)*sin(alpha),   cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                  [0,0,0,1]])

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
    ur5e_StandParam_path = home + 'DH_Parameter_Standard_UR5.csv'
    ur5e_ModifParam_path = home + 'DH_Parameter_Modified_UR5.csv'
    

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


    for i in reader_st:
        M = createStandardDHMatrix(float(i[0]), float(i[1]), float(i[2]), 0.0)
        Standard_Mat = Standard_Mat * M

    for i in reader_md:   
        M = createModifiedDHMatrix(float(i[0]), float(i[1]), float(i[2]), 0.0)
        Modified_Mat = Modified_Mat * M

    print("Standard : ", Standard_Mat)
    print("Modified : ", Modified_Mat)

   # M = createModifiedDHMatrix(0.0, 0.0, 1.57, 1.0)* createModifiedDHMatrix(0.0, 0.5, 1.57, 0.0) * createModifiedDHMatrix(0.0, 1.5, 0.0, -0.2)
    #print(M)

    #M2 = createStandardDHMatrix(0.0, 0.0, 1.57, 1.0)*createStandardDHMatrix(0.0, 0.5, 1.57, 0.0)*createStandardDHMatrix(0.0, 1.5, 0.0, -0.2)
    #print(M2)

if __name__ == "__main__":
    main()

    x = Symbol('x2')
    y=Symbol('y') 
    expr=x**2+y**2
    print(expr) 



