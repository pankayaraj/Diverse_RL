import numpy as np


class Transition_tuple():

    def __init__(self, state, action, reward, next_state,
                 inital_state, time_step
                 ):
        #expects as list of items for each initalization variable
        self.state = np.array(state)
        self.action = np.array(action)
        self.reward = np.array(reward)
        self.next_state = np.array(next_state)
        self.initial_state = np.array(inital_state)
        self.time_step = np.array(time_step)


    def get_all_attributes(self):
        return [self.state, self.action, self.reward, self.next_state, self.initial_state, self.time_step]

class Last_Episode_Container():

    def __init__(self):
        self.no_data = 0
        self.position = 0


        #complete ones
        self.state, self.action, self.reward, self.next_state, self.inital_state, self.time_step = [], [], [], [], [], []
        #incomplete ones
        self.incomplete_state, self.incomplete_action, self.incomplete_reward, self.incomplete_next_state, self.incomplete_inital_state, self.incomplete_time_step = [], [], [], [], [], []

    def push(self, state, action, reward, next_state, inital_state, time_step, end_of_eps = False):

        if end_of_eps != True:
            self.incomplete_state.append(state)
            self.incomplete_action.append(action)
            self.incomplete_reward.append(reward)
            self.incomplete_next_state.append(next_state)
            self.incomplete_inital_state.append(inital_state)
            self.incomplete_time_step.append(time_step)

            self.no_data += 1

            self.position = (self.position + 1)


        else:

            self.state, self.action, self.reward, self.next_state, self.inital_state, self.time_step = self.incomplete_state, self.incomplete_action, self.incomplete_reward, self.incomplete_next_state, self.incomplete_inital_state, self.incomplete_time_step
            self.incomplete_state, self.incomplete_action, self.incomplete_reward, self.incomplete_next_state, self.incomplete_inital_state, selfincomplete_time_step = [], [], [], [], [], []

    def sample(self):

        if len(self.state) == 0:
            raise AttributeError("Episodes Not Pusehd. Initalize !")

        state = np.array(self.state)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_state = np.array(self.next_state)
        initial_state = np.array(self.inital_state)
        time_step = np.array(self.time_step)

        return Transition_tuple(state, action, reward, next_state, initial_state, time_step)

    def iterate_through(self):
        all_data = self.sample()
        all_attributes = all_data.get_all_attributes()

        for i in range(self.no_data):
            t = []
            for j in range(len(all_attributes)):
                t.append(all_attributes[j][i])

            yield t

    def __len__(self):
        return len(self.state)