import numpy as np
import heapq
import random
from itertools import count

class Transition_tuple():

    def __init__(self, state, action, reward, next_state,
                 inital_state, time_step, optim_traj
                 ):
        #expects as list of items for each initalization variable
        self.state = np.array(state)
        self.action = np.array(action)
        self.reward = np.array(reward)
        self.next_state = np.array(next_state)
        self.initial_state = np.array(inital_state)
        self.time_step = np.array(time_step)
        self.optim_traj = np.array(optim_traj)


    def get_all_attributes(self):
        return [self.state, self.action, self.reward, self.next_state, self.initial_state, self.time_step, self.optim_traj]

class Replay_Memory():

    def __init__(self, capacity=10000):
        self.no_data = 0
        self.position = 0
        self.capacity = capacity
        self.state, self.action, self.reward, self.next_state, self.inital_state, self.time_step, self.optim_traj = [], [], [], [], [], [], []
        self.tiebreaker = count()
    def push(self, state, action, reward, next_state, inital_state, time_step, optim_traj, priority):


        p = priority
        c = next(self.tiebreaker)


        if len(self.state) < self.capacity:

            heapq.heappush(self.state, (p,c, state))
            heapq.heappush(self.action, (p,c, action))
            heapq.heappush(self.reward, (p,c, reward))
            heapq.heappush(self.next_state, (p,c, next_state))
            heapq.heappush(self.inital_state, (p,c, inital_state))
            heapq.heappush(self.time_step, (p,c, time_step))
            heapq.heappush(self.optim_traj, (p,c, optim_traj))
            self.no_data += 1

        heapq.heapreplace(self.state, (p,c, state))
        heapq.heapreplace(self.action, (p,c, action))
        heapq.heapreplace(self.reward, (p,c, reward))
        heapq.heapreplace(self.next_state, (p,c, next_state))
        heapq.heapreplace(self.inital_state, (p,c, inital_state))
        heapq.heapreplace(self.time_step, (p,c, time_step))
        heapq.heapreplace(self.optim_traj, (p,c, optim_traj))


        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        state, action, reward, next_state, initial_state, time_step, optim_traj = [], [], [], [], [], [], []


        if len(self.state) < self.capacity:
            indices = np.random.choice(len(self.state), batch_size)
        else:
            indices = np.random.choice(self.capacity, batch_size)

        for index in indices:
            state.append(self.state[index][2])
            action.append(self.action[index][2])
            reward.append(self.reward[index][2])
            next_state.append(self.next_state[index][2])
            initial_state.append(self.inital_state[index][2])
            time_step.append(self.time_step[index][2])
            optim_traj.append(self.optim_traj[index][2])

        return Transition_tuple(state, action, reward, next_state, initial_state, time_step, optim_traj)

    def iterate_through(self):
        all_data = self.sample(self.no_data)
        all_attributes = all_data.get_all_attributes()

        for i in range(self.no_data):
            t = []
            for j in range(len(all_attributes)):
                t.append(all_attributes[j][i])

            yield t

    def __len__(self):
        return len(self.state)


def combine_transition_tuples(tuples):


    state = np.array([])
    action = np.array([])
    reward = np.array([])
    next_state = np.array([])
    initial_state = np.array([])
    time_step = np.array([])
    optim_traj = np.array([])

    i = 0
    for tuple in tuples:

        s, a, r, n_s, i_s, t, o_t = tuple.state, tuple.action, tuple.reward, tuple.next_state, tuple.initial_state, tuple.time_step, tuple.optim_traj
        if i == 0:
            state = s
            action = a
            reward = r
            next_state = n_s
            initial_state = i_s
            time_step = t
            optim_traj = o_t
        else:


            state = np.concatenate((state, s), axis=0)
            action = np.concatenate((action, a), axis=0)
            reward = np.concatenate((reward, r), axis=0)
            next_state = np.concatenate((next_state, n_s), axis=0)
            initial_state = np.concatenate((initial_state, i_s), axis=0)
            time_step = np.concatenate((time_step, t), axis=0)
            optim_traj = np.concatenate((optim_traj, o_t), axis=0)

        i += 1

    main_tuple = Transition_tuple(state, action, reward, next_state, initial_state, time_step, optim_traj)
    return main_tuple


