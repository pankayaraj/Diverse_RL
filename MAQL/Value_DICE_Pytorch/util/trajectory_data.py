import numpy as np


class Transition_tuple():

    def __init__(self, state, action, reward, next_state,
                 time_step, discount, action_prob, inital_state
                 ):
        #expects as list of items for each initalization variable


        self.state = np.array(state)
        self.action = np.array(action)
        self.reward = np.array(reward)
        self.next_state = np.array(next_state)
        self.time_step = np.array(time_step)
        self.discount = np.array(discount)
        self.action_prob = np.array(action_prob)
        self.initial_state = np.array(inital_state)


    def get_all_attributes(self):
        return [self.state, self.action, self.reward, self.next_state, self.time_step, self.discount, self.action_prob, self.initial_state]

class Trajectory_data():

    def __init__(self, trajectories, policy=None):

        """
        Trajectories is a list of list of tuples (state, action, reward, next_state)
        """
        self.trajectories = trajectories
        self.policy = policy
        self.no_data = 0

        self.state, self.action, self.reward, self.next_state, self.time_step, self.discount, self.action_prob, self.inital_state = [], [], [], [], [], [], [], []


        for trajectory in trajectories:

            initial_state, _ , _, _ = trajectory[0]

            for step, (state, action, reward, next_state) in enumerate(trajectory):

                self.state.append(state)
                self.action.append(action)
                self.reward.append(reward)
                self.next_state.append(next_state)
                self.time_step.append(step)

                # considered as the infinite horizon problem
                self.discount.append(1.0)
                if self.policy == None:
                    self.action_prob.append(None)
                else:
                    a_n = 0
                    for i in range(np.shape(action)[0]):
                        if action[i] == 1:
                            a_n = i

                    self.action_prob.append(self.policy.get_probability(np.array([state]), a_n, format="numpy")[0])

                self.inital_state.append(initial_state)

                self.data = Transition_tuple(state=self.state, action=self.action, reward=self.reward, next_state=self.next_state,
                                             time_step=self.time_step, discount=self.discount, action_prob=self.action_prob, inital_state=self.inital_state)

                self.no_data += 1

    def sample(self, size):

        indices = np.random.choice(self.no_data, size)

        sample = []
        for attribute in self.data.get_all_attributes():
            sample.append(np.take(attribute, indices, axis=0))

        return Transition_tuple(*sample)

    def iterate_through(self):
        all_attributes = self.data.get_all_attributes()

        for i in range(self.no_data):
            t = []
            for j in range(len(all_attributes)):
                t.append(all_attributes[j][i])

            yield t

