import matplotlib.pyplot as plt
import numpy  as np
from scipy.spatial import Voronoi, voronoi_plot_2d
 

x = np.linspace(0, 10, 100)
y = x + np.random.randn(100) 

points = np.random.random((10, 2))

xx = []
yy = []

for a in points:
    xx.append(a[0])
    yy.append(a[1])

plt.scatter(xx,yy)


vor = Voronoi(points)
plt.figure(figsize=(6, 4), facecolor='white')
ax = plt.subplot(aspect='equal')
voronoi_plot_2d(vor, ax=ax)

plt.show()