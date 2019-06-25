import os 
import numpy as np
from matplotlib import pyplot as plt

O = np.array([0,0])

x = np.arange(5)
y = (1,2,3,4,5)




width = 0.3
yerr = (.1, .08, .1, .0, .5)

#plt.bar(x, y, width, align='center',yerr=yerr, ecolor='r')
plt.scatter(O[0],O[1])

plt.show()


