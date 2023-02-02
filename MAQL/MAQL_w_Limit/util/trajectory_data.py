import numpy as np


class Transition_tuple():

    def __init__(self, state, action, next_state,
                 inital_state, time_step
                 ):
        #expects as list of items for each initalization variable
        self.state = np.array(state)
        self.action = np.array(action)
        self.next_state = np.array(next_state)
        self.initial_state = np.array(inital_state)
        self.time_step = np.array(time_step)


    def get_all_attributes(self):
        return [self.state, self.action, self.next_state, self.initial_state, self.time_step]

class Trajectory_data():

    def __init__(self, capacity=10000):
        self.no_data = 0
        self.position = 0
        self.capacity = capacity
        self.state, self.action, self.next_state, self.inital_state, self.time_step = [], [], [], [], []

    def push(self, state, action, next_state, inital_state, time_step):
        if len(self.state) < self.capacity:
            self.state.append(None)
            self.action.append(None)
            self.next_state.append(None)
            self.inital_state.append(None)
            self.time_step.append(None)

            self.no_data += 1


        self.state[self.position] = state
        self.action[self.position] = action
        self.next_state[self.position] = next_state
        self.inital_state[self.position] = inital_state
        self.time_step[self.position] = time_step

        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):

        if len(self.state) < self.capacity:
            indices = np.random.choice(len(self.state), batch_size)
        else:
            indices = np.random.choice(self.capacity, batch_size)

        state = np.take(np.array(self.state), indices, axis=0)
        action = np.take(np.array(self.action), indices, axis=0)
        next_state = np.take(np.array(self.next_state), indices, axis=0)
        initial_state = np.take(np.array(self.inital_state), indices, axis=0)
        time_step = np.take(np.array(self.time_step), indices, axis=0)

        return Transition_tuple(state, action, next_state, initial_state, time_step)

    def iterate_through(self):
        all_data = self.sample(self.no_data)
        all_attributes = all_data.get_all_attributes()

        for i in range(self.no_data):
            t = []
            for j in range(len(all_attributes)):
                t.append(all_attributes[j][i])

            yield t
