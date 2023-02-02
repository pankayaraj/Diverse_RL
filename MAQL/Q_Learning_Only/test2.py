import numpy as np
import torch
import gym
from env.gridworld.environments import GridWalk


from q_learning import Q_learning

from model import NN_Paramters
from parameters import Algo_Param, Save_Paths, Load_Paths

q_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.01)
algo_param = Algo_Param()
algo_param.gamma = 0.995

#env = gym.make("CartPole-v0")
grid_size = 10
env = GridWalk(grid_size, False)

Q = Q_learning(env, q_param, algo_param)
Q.load("q", "target_q")


state = env.reset()

print(state)
rew = 0
for i in range(100):

    action = Q.get_action(state)

    state, reward, done, _ = env.step(action)
    rew += reward
    if done == True:
        break
    #env.render()
print(state)
print(rew)