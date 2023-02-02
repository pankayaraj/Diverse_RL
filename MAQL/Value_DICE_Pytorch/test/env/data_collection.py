from env.gridworld.environments import GridWalk
import numpy as np
from env.gridworld.optimal_policy import Optim_Policy_Gridwalk
from util.collect_trajectories import collect_data

env = GridWalk(5, tabular_obs=False)


s = np.array([1,1])
s_ = np.array([2,2])
s__ = np.array([3,3])

trajecotry = [(s, None, None, None), (s_, None, None, None), (s__, None, None, None)]
#env.render_trajectory(trajecotry)


P = Optim_Policy_Gridwalk(grid_env=env)
data, _, _ = collect_data(env, P, 1, 10)

T = []
for i in data.iterate_through():
    print(i)
    T.append(i[0:4])

env.render_trajectory(T)

