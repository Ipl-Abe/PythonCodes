import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


g = 9.8 # Gravitational acceleration [m/s^2]
L = 10 # Distance between monkey and hunter[m]
theta = np.pi/4 # Firing angle for bullet [rad]
v0 = 10 # Initial velocity for the bullet [m/s]
interval = 10 # Calculation time interval [ms]

t = np.arange(0,L/v0,interval/1000.)
#t = np.arange(0,5,1/2.)
y0 = [[0,v0*np.sin(theta)], #Initial condition for the bullet (y,v) 
      [L*np.sin(theta),0]]  #Initial condition for the bullet (y,v)

def equation(y,t,g): 
    ret = [y[1],-1*g] 
    return ret

y1 = odeint(equation,y0[0],t,args=(g,))
y2 = odeint(equation,y0[1],t,args=(g,))

#y1.T
# debug coordinate data
# print(y1.T)

fig,ax = plt.subplots()
obj1, = ax.plot([],[],'o')
obj2, = ax.plot([],[],'^')
ax.set_xlim(0,L*np.cos(theta)*1.2)
ax.set_ylim(min(y1.T[0])*1.2,L*np.sin(theta)*1.2)
ax.set_aspect('equal')
ax.set_title('v0={},theta={}°'.format(v0,theta*180/np.pi))

def update_anim(frame_num):
    obj1.set_data(v0*np.cos(theta)*t[frame_num],y1.T[0][frame_num]) #(水平方向の速度×経過時間, 鉛直方向の位置)
    obj2.set_data(L*np.cos(theta),y2.T[0][frame_num])
    return obj1, obj2,

anim = FuncAnimation(fig,update_anim,frames=np.arange(0,len(t)),interval=interval,blit=True,repeat=True)
plt.show()

