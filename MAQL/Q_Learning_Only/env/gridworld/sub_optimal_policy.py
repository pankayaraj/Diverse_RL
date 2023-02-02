import torch
import numpy as np

class Sub_Optim_Policy_Gridwalk():

    """
    This is the optimal policy for a gridwalk problem where the goal is loacated at the corner

    """
    def __init__(self, grid_env, action_dim,  eps_explore = 0.0):
        self.grid_env = grid_env
        self.eps_explore = eps_explore
        self.action_dim = action_dim

    def forward(self, state):
        # Optimal policy takes shortest path to (max_x, max_y) point in grid.

        state = self.grid_env.get_tabular_obs(state[0])

        xy = self.grid_env.get_xy_obs(state)
        x, y = xy[Ellipsis, 0], xy[Ellipsis, 1]
        actions = np.where(x >= y, 0, 1)  # Increase x or increase y by 1.

        probs = np.zeros((actions.size, 4))
        probs[np.arange(actions.size), actions] = 1
        probs = probs.reshape(list(actions.shape) + [4])

        return (probs * (1 - self.eps_explore) +
                0.25 * np.ones([4]) * self.eps_explore)

    def sample(self, state, format="numpy"):
        # returns a one hot vector as numpy array

        action_prob = self.forward(state)
        sample = np.random.multinomial(1, action_prob)  # slightly abusing the syntax

        a = [0.0 for i in range(self.action_dim)]
        for i in range(len(sample)):
            if sample[i] == 1:
                a[i] = 1
        return np.array([a])

    def get_probablities(self, state, format="numpy"):
        return np.reshape(self.forward(state), newshape=(self.grid_env._n_action, 1))

    def get_probability(self, state, action_n, format="numpy"):
        action_scaler = action_n
        return np.reshape(self.forward(state)[action_scaler], newshape=(1, 1))

    def get_log_probability(self, state, action_n, format="numpy"):
        return np.log(1e-8 + self.get_probability(state, action_n))