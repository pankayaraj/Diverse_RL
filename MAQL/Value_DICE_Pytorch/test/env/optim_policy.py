from env.gridworld.environments import GridWalk
from env.gridworld.optimal_policy import Optim_Policy_Gridwalk
import numpy as np


G = GridWalk(10)

P = Optim_Policy_Gridwalk(G)

state = np.array([1, 2])

print(P.get_probability(state, 0))
print(P.get_log_probability(state, 0))