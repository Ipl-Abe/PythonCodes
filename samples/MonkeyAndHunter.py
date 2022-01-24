import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.8 # Gravitational acceleration [m/s^2]
L = 10 # Distance between monkey and hunter[m]
theta = np.pi/4 # Firing angle for bullet [rad]
v0 = 10 # Initial velocity for the bullet [10m/s]
interval = 10 # Calculation time interval [ms]

t = np.arange(0,L/v0,interval/1000)
y0 = [[0,v0*np.sin(theta)], #Initial condition for the bullet (y,v) 
      [L*np.sin(theta),0]]  #Initial condition for the bullet (y,v)

def equation(y,t,g): 
    ret = [y[1],-1*g] 
    return ret

y1 = odeint(equation,y0[0],t,args=(g,))
y2 = odeint(equation,y0[1],t,args=(g,))