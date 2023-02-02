import numpy as np
import random


class Transition_tuple_Q_L():

    def __init__(self, state, action, reward, next_state, done
                 ):
        #expects as list of items for each initalization variable
        self.state = np.array(state)
        self.action = np.array(action)
        self.reward = np.array(reward)
        self.next_state = np.array(next_state)
        self.done = np.array(done)

    def get_all_attributes(self):
        return [self.state, self.action, self.reward, self.next_state, self.done]

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.state, self.action, self.reward, self.next_state, self.done = [], [], [], [], []
        self.position = 0


    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        if len(self.state) < self.capacity:
            self.state.append(None)
            self.action.append(None)
            self.reward.append(None)
            self.next_state.append(None)
            self.done.append(None)

        self.state[self.position] = state
        self.action[self.position] = np.array([action])
        self.reward[self.position] = np.array([reward])
        self.next_state[self.position] = next_state
        self.done[self.position] = np.array([done])

        self.position = (self.position + 1) % self.capacity

    def sample(self, size):

        if len(self.state) < self.capacity:
            indices = np.random.choice(len(self.state), size)
        else:
            indices = np.random.choice(self.capacity, size)

        state = np.take(np.array(self.state), indices, axis=0)
        action = np.take(np.array(self.action), indices, axis=0)
        reward = np.take(np.array(self.reward), indices, axis=0)
        next_state= np.take(np.array(self.next_state), indices, axis=0)
        done = np.take(np.array(self.done), indices, axis=0)
        return Transition_tuple_Q_L(state, action, reward, next_state, done)

    def __len__(self):
        return len(self.state)


