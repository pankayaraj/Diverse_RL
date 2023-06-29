import matplotlib.pyplot as plt
import numpy as np
from env.gridworld.environments_obs_1 import GridWalk
from matplotlib import colors


env = GridWalk(10, False)
env.reset()

grid = [[5 for i in range(10)] for j in range(10)]
for o in env.obstacle:
    grid[o[1]][o[0]] = 3

grid[env._y][env._x] = 2
grid[env._target_y][env._target_x] = 4



cmap = colors.ListedColormap(['red', 'white', "black", "green", "blue", "yellow", "grey"])
bounds = [-2, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
norm = colors.BoundaryNorm(bounds, cmap.N)


fig, ax = plt.subplots()
ax.imshow(grid, cmap=cmap, norm=norm)




# draw gridlines
ax.grid(which='major', axis='both', linestyle=' ', color='k', linewidth=2, )
ax.legend=['red', 'white', "black", "green", "blue", "yellow", "grey"]
ax.set_xticks(np.arange(-0.5, 10.5, 1))
ax.set_yticks(np.arange(-0.5, 10.5, 1))

plt.savefig("gradual_models/10x10/grid_obs_2.png")