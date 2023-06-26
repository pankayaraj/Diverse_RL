import torch





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import sys


no = 0
grid1 = torch.load( sys.path[0] + "/occ_1_" + str(no))
grid2 = torch.load(sys.path[0] + "/occ_2_" + str(no))
# Create a 2D array

print(grid1)
print(np.array(grid2)/np.sum(grid2))

data = [[0 for i in range(10)] for j in range(10)]

for i in range(10):
    for j in range(10):
        data[i][j] = grid1[i][j]/(grid2[i][j]+0.01)
data = np.array(data)

data = data/np.sum(data)

# Define x and y coordinates
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])

# Create a meshgrid from x and y
X, Y = np.meshgrid(x, y)

# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



# Plot the surface with a colormap
surf = ax.plot_surface(X, Y, data, cmap='RdYlBu_r')

# Add colorbar and set labels
fig.colorbar(surf)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Occupancy Measure')
ax.set_zlim(0,0.05)

ax.set_title("Ratio_" + str(no), fontsize = 30)
# Show the plot
plt.savefig(sys.path[0] + "/ratio_"+ str(no) + ".png")
