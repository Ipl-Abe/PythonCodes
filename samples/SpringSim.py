import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import ArrowStyle

interval = 1 #計算上の時間間隔[ms]
t = np.arange(0,20,interval/1000)

k = 50 #バネ定数[N/m]
m = 1.0 #重りの質量[kg]
w = np.sqrt(k/m)
w0 = np.sqrt(k/m) #今回は共振を再現するためにw=w0
x0 = [0,-1*w**2*0.2] #初期条件(v,x). つまり重りを釣り合いの位置から20cm離れたところからスタート
f = 0 #外力の大きさ
a = 2 #減衰係数

def equation(x,t,w,f,m,w0,a):
    ret = [-1*w**2 * x[1] + f/m*np.cos(w0*t) - a/m*x[0], x[0]]
    return ret

x = odeint(equation,x0,t,args=(w,f,m,w0,a))

fig,ax = plt.subplots()
image, = ax.plot([],[], 'o', lw=2)
arrow, = ax.plot([],[], 'o', lw=2)

ax.plot()
ax.set_xlim(min(x.T[1]*(-1)*m/k)*1.2,max(x.T[1]*(-1)*m/k)*1.2)
ax.set_ylim(-0.1,0.1)
ax.set_title('f={},a={}'.format(f,a))
ax.grid(axis='x')

def update_anim(frame_num):
#    print(x.T[1][frame_num])
    rect = Rectangle((x.T[1][frame_num]*(-1)*m/k - 0.025, -0.025),0.05, 0.05,linewidth=1,edgecolor='r',facecolor='none')
    image = ax.add_patch(rect)
    arrow = ax.annotate('', xy=(0,0), xytext=(x.T[1][frame_num]*(-1)*m/k,0),
                    arrowprops=dict(arrowstyle=ArrowStyle('->', head_length=1, head_width=1),
                                    facecolor='C0',
                                    edgecolor='C0')
                    )
    
    return image, arrow

anim = FuncAnimation(fig, update_anim,frames=np.arange(0, len(t)),interval=interval ,blit=True)
plt.show()