import torch
import numpy as np

class Optim_Policy_Gridwalk():

    """
    This is the optimal policy for a gridwalk problem where the goal is loacated at the corner
    action description
        0 : right, 1: left, 2:up, 3:down
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

        #actions = np.where(x <= y, 0, 1)  # Increase x or increase y by 1.


        delta_x = self.grid_env._target_x - x
        delta_y = self.grid_env._target_y - y

        if abs(delta_x) > abs(delta_y):
            if delta_x > 0:
                actions = 0
            elif delta_x < 0:
                actions = 1
            else:
                actions = 4
        else:
            if delta_y > 0:
                actions = 2
            elif delta_y < 0:
                actions = 3
            else:
                actions = 4

        probs = np.zeros((1, 5))
        probs[0, actions] = 1
        probs = probs.reshape([5])
        return (probs * (1 - self.eps_explore) +
                0.2 * np.ones([5]) * self.eps_explore)

    def sample(self, state, format="numpy"):
        #returns a one hot vector as numpy array
        if np.shape(state)[0] != 1:
            #batched scenario
            a_ = []
            for i in range(np.shape(state)[0]):
                action_prob = self.forward(state)
                sample = np.random.multinomial(1, action_prob)  # slightly abusing the syntax

                a = [0.0 for i in range(self.action_dim)]
                for i in range(len(sample)):
                    if sample[i] == 1:
                        a[i] = 1
                a_.append(a)
            return np.array(a_)
        else:
            action_prob = self.forward(state)
            sample = np.random.multinomial(1, action_prob) #slightly abusing the syntax

            a = [0.0 for i in range(self.action_dim)]
            for i in range(len(sample)):
                if sample[i] == 1:
                    a[i] = 1
            return np.array([a])


    def get_probabilities(self, state, format="numpy"):
        return np.reshape(self.forward(state), newshape=(self.grid_env._n_action, 1))

    def get_probability(self, state, action_n, format="numpy"):
        action_scaler = action_n
        return np.reshape(self.forward(state)[action_scaler], newshape=(1,1))

    def get_log_probability(self, state, action_n, format="numpy"):
        return np.log(1e-8 + self.get_probability(state, action_n))