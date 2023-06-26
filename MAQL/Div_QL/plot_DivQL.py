import torch
from DivQL_main import DivQL
from env.gridworld.environments import GridWalk
from model import NN_Paramters
from parameters import Algo_Param, Save_Paths, Load_Paths


q_param = NN_Paramters(state_dim=2, action_dim=4, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.05)
nu_param = NN_Paramters(state_dim=2, action_dim=4, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.0001)
algo_param = Algo_Param()
algo_param.gamma = 0.9


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

env = GridWalk(10, False)

inital_log_buffer = torch.load("gradual_models/10x10/q/mem0")


M  = DivQL(env, inital_log_buffer=inital_log_buffer, q_nn_param=q_param, nu_pram=nu_param,
           algo_param=algo_param, num_z=15)


M.load_main( "gradual_models/10x10/DivQL/q1", "gradual_models/10x10/DivQL/target_q1" )

z_no = 8

for z in range(z_no):
    data = [[0 for i in range(10)] for j in range(10)]
    for i in range(z):
        for x in range(10):
            s = env.reset()
            i_s = s
            rew = 0
            data[s[1]][s[0]] += 1
            S = []

            for j in range(100):
                a = M.get_action(s,i)
                s, r, d, _ = env.step(a)
                S.append(s)

                rew += r
                data[s[1]][s[0]] += 1

                if j == 100 - 1:
                    d = True
                if d == True:
                    break
        print(rew, S)




    data = data/np.sum(data)
    print(data)

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


    ax.set_title("DivQL, Ratio, Tabular", fontsize = 30)
    # Show the plot
    plt.savefig("gradual_models/10x10/DivQL_ratio_z")
