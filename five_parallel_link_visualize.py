import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sympy import *
from sympy.abc import *


def norm2(x):
    return x[0]*x[0]+x[1]*x[1]

def symbol_calc():
    angles = symbols("phi_0 phi_1 ")
    thetas = symbols("theta_0 theta_1")

    #unit_vectors = np.array([[1,0],[-1,0]])
    unit_vectors = [Matrix([cos(angles[i]), sin(angles[i])]) for i in range(2)]
    #unit_vectors2 = Matrix([[1],[0]],[[-1],[0]])
    
    print(unit_vectors)
    #print(unit_vectors2)
    params = [(A,1.0),(B,2.0)]
    params.append((angles[0],0))
    params.append((angles[1],pi))

    params.append((x,0))    
    params.append((y,1))
    ey = Matrix([0,1])
    
    #A_vectors = A*unit_vectors[0]
    #A_vectors = map(lambda x: A*x, unit_vectors)
    A_vectors = [A*unit_vectors[0], A*unit_vectors[1]]
    #unit_vectors
    #print(unit_vectors[0])
    print(A_vectors)

    print("Debug")

    B_vectors = [A_vectors[i] + B*(unit_vectors[i]*cos(thetas[i]) + ey * sin(thetas[0])) for i in range(2)]
    print(B_vectors)
    print("Debug2")
   
    C_vectors = Matrix([x,y])
    print(C_vectors)
    #print(A_vectors -B_vectors)
    #UU = C_vectors - B
    print(norm2((C_vectors-B_vectors[0])))
    eq0 = C**2 - norm2((C_vectors-B_vectors[0]))
    print(eq0)

    eq0 = expand(eq0)
    print(eq0)
    #eq0 = simplify(eq0)
    print("Simplify")
    print(eq0)


    # Next Step
    formula = Eq(P +Q * sin(thetas[0]) + R * cos(thetas[0]),0)
    print(formula)


    thetas_ans = solve(formula,thetas[0])
    print(thetas_ans)

    Q_dash = eq0.coeff(sin(thetas[0]),1)
    R_dash = eq0.coeff(cos(thetas[0]),1)
    P_dash = eq0.coeff(cos(thetas[0]),0)   #- expand(Q_dash * sin(thetas[0]))
    print("Q_dash : ")
    print(Q_dash)
    print("R_dash : ")
    print(R_dash)
    print("P_dash : ")
    print(P_dash)

    print("Q_dash test : ")
    print(expand(Q_dash * sin(thetas[0])))
    


    return 1




if __name__ == "__main__":
    symbol_calc()
    print("Fin")




