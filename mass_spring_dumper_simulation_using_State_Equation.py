import numpy as np
import control.matlab as ctrl
import matplotlib.pyplot as plt

m = 1    # 質量 [kg] 非負
c = 1    # 減衰係数 [N/m] 非負
k = 10   # バネ係数 [Ns/m] 非負

sys = ctrl.tf((1),(m,c,k)) # 伝達関数
print(sys)

t_range = (0,10) # 0～10秒の範囲をシミュレーション
y, t = ctrl.impulse(sys, T=np.arange(*t_range, 0.01))

print(y)

plt.figure(figsize=(7,4),dpi=120,facecolor='white')
plt.hlines(0,*t_range,colors='gray',ls=':')
plt.plot(t,y)
plt.xlim(*t_range)
plt.show()