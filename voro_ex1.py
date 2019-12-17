import matplotlib.pyplot as plt
import numpy  as np
from scipy.spatial import Voronoi, voronoi_plot_2d
 
points = np.random.random((10, 2))
vor = Voronoi(points)
plt.figure(figsize=(6, 4), facecolor='white')
ax = plt.subplot(aspect='equal')
voronoi_plot_2d(vor, ax=ax)
plt.show()