import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter 
from matplotlib.patches import Rectangle
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import ArrowStyle
import os 

from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))

interval = 2 #Interval for calculation[ms]
t = np.arange(0,1,interval/1000)

k = 50 #Coeficient for friction [N/m]
m = 1.0 # weight[kg]
w = np.sqrt(k/m)
w0 = np.sqrt(k/m) # w=w0
x0 = [0,-1*w**2*0.2] #Initial condition(v,x). start from 0.2 cm
f = 0 # External force
a = 0 # Attenuation

def equation(x,t,w,f,m,w0,a):
    ret = [-1*w**2 * x[1] + f/m*np.cos(w0*t) - a/m*x[0], x[0]]
    return ret

x = odeint(equation,x0,t,args=(w,f,m,w0,a))

fig,ax = plt.subplots()
image, = ax.plot([],[], 'o', lw=2)
arrow, = ax.plot([],[], 'o', lw=2)
box_center, = ax.plot([],[], 'o', lw=2)

major_ticks_top=np.linspace(0,20,6)
minor_ticks_top=np.linspace(0,20,21)

ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-0.20, 0.20)
ax.set_xticks(major_ticks_top)
ax.set_title('External force={}, Attenuation={}'.format(f,a))
ax.grid(axis='x')

def update_anim(frame_num):
    ax.cla()
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    major_ticks_top=np.linspace(-0.25,0.25,11)
    ax.set_xticks(major_ticks_top)
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.20, 0.20)
    ax.set_title('External force={}, Attenuation={}'.format(f,a))
    ax.grid(axis='x')

    rect = Rectangle((x.T[1][frame_num]*(-1)*m/k - 0.025, -0.025),0.05, 0.05,linewidth=1,edgecolor='r',facecolor='none')
    image = ax.add_patch(rect)
    circle = patches.Circle(xy=(x.T[1][frame_num]*(-1)*m/k, 0.0), radius=0.005, fc='g', ec='r')
    box_center = ax.add_patch(circle)
    arrow = ax.annotate('', xy=(0,0), xytext=(x.T[1][frame_num]*(-1)*m/k,0),
                    arrowprops=dict(arrowstyle=ArrowStyle('->', head_length=1, head_width=1),
                                    facecolor='C0',
                                    edgecolor='C0')
                    )
    print(frame_num)
    return image, box_center, arrow


fpath_string=dir_path+'pillow_imagedraw.gif'
anim = FuncAnimation(fig, update_anim,frames=np.arange(0, len(t)),interval=interval ,blit=True)
writer = PillowWriter(fps=10)
anim.save(fpath_string, writer=writer)
plt.show()
