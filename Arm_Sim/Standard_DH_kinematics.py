import numpy
from sympy import *
import sys
import os


def createModifiedDHMatrix(alpha, a, q, d):
    mat = Matrix([[           cos(q),             -sin(q),           0,             a],
                  [sin(q)*cos(alpha),   cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                  [sin(q)*sin(alpha),   cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                  [                0,                   0,           0,             1]])

    return mat

def createStandardDHMatrix(alpha, a, q, d):
    mat = Matrix([[      cos(q),  -sin(q)*cos(alpha),   sin(q)*sin(alpha),  cos(q)*a],
                  [      sin(q),   cos(q)*cos(alpha),  -cos(q)*sin(alpha),  sin(q)*a],
                  [           0,          sin(alpha),          cos(alpha),         d],
                  [           0,                   0,                   0,         1]])

    return mat
    


def main():



    M = createModifiedDHMatrix(0.0, 0.0, 1.57, 1.0)* createModifiedDHMatrix(0.0, 0.5, 1.57, 0.0) * createModifiedDHMatrix(0.0, 1.5, 0.0, -0.2)
    print(M)

    M2 = createStandardDHMatrix(0.0, 0.0, 1.57, 1.0)*createStandardDHMatrix(0.0, 0.5, 1.57, 0.0)*createStandardDHMatrix(0.0, 1.5, 0.0, -0.2)
    print(M2)

if __name__ == "__main__":
    main()



