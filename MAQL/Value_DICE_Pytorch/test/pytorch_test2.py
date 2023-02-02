import torch
from gym_minigrid.wrappers import *
import numpy as np


env = gym.make('MiniGrid-CustomCrossingS9N1-v0')
env = FullyObsWrapper(env) # Get pixel observations
obs = env.reset() # This now produces an RGB tensor only

for i in range(1000):
    print(i, obs)
    action= np.random.randint(0, env.action_space.n-1)
    obs  = env.step(action)
    env.render('human')