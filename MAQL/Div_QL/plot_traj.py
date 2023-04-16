import matplotlib.pyplot as plt
import numpy as np
import torch

n =  4

file = {4:"1", 6:"2", 10:"3"}
Trajectories = torch.load("gradual_models/" + file[n] + "/Trajectories")

Grid = [[0 for i in range(10)] for j in range(10)]
Grid[0][0] = 1
Grid[9][9] = 2
X, Y = [], []
for i in range(1, len(Trajectories)+1):
    x_ = []
    y_ = []
    Traj = Trajectories[i-1]

    for s in Traj:
        x = int(s[0])
        y = int(s[1])
        x_.append(x)
        y_.append(y)

        #Grid[x][y] = i
    X.append(x_)
    Y.append(y_)

fig, ax = plt.subplots()
for i in range(len(Trajectories)):
    x = X[i]
    y = Y[i]
    ax.plot(x,y)

    arrow_x = np.array(x[:-1])
    arrow_y = np.array(y[:-1])
    dx = np.array(x[1:]) - np.array(x[:-1])
    dy = np.array(y[1:]) - np.array(y[:-1])

    # Plot the arrows on the axis object
    ax.quiver(arrow_x, arrow_y, dx, dy, angles='xy', scale_units='xy', scale=1, color='g')

ax.imshow(Grid,)
plt.title("z = " + str(n))
plt.savefig("gradual_models/" + str(n) +".png")