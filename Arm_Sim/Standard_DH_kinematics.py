import numpy
from sympy import *
import sys
import os
import csv



def createModifiedDHMatrix(alpha, a, d, q):
    mat = Matrix([[           cos(q),             -sin(q),           0,             a],
                  [sin(q)*cos(alpha),   cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                  [sin(q)*sin(alpha),   cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                  [                0,                   0,           0,             1]])

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
    #print(home)
    path = home + 'DH_Parameter.csv'
    f = open(path)
    reader = csv.reader(f)
    header = next(reader)
    
    Modified_Mat = createModifiedDHMatrix(0.0, 0.0, 0.0, 0.0)
    #Standard_Mat = createStandardDHMatrix(0.0, 0.0, 0.0, 0.0)
    l = [row for row in reader]
    print(l)

    # readerの行数が１つ多く読み込まれる
    for i in reader:
        print(i[0],i[1],i[2],i[3])    
        #M1 = createModifiedDHMatrix(i[0],i[1],i[2], 0.0)
        #Modified_Mat = Modified_Mat * M1
        #M2 = createStandardDHMatrix(i[0],i[1],i[2], 0.0)
        #Standard_Mat = Standard_Mat * M2
    #print(Standard_Mat)
    #print(Modified_Mat)

   # M = createModifiedDHMatrix(0.0, 0.0, 1.57, 1.0)* createModifiedDHMatrix(0.0, 0.5, 1.57, 0.0) * createModifiedDHMatrix(0.0, 1.5, 0.0, -0.2)
    #print(M)

    #M2 = createStandardDHMatrix(0.0, 0.0, 1.57, 1.0)*createStandardDHMatrix(0.0, 0.5, 1.57, 0.0)*createStandardDHMatrix(0.0, 1.5, 0.0, -0.2)
    #print(M2)

if __name__ == "__main__":
    main()



