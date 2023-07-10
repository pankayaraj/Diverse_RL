import numpy as np


class Transition_tuple():

    def __init__(self, state, action, action_mean, reward, next_state,
                 inital_state, done, time_step, optim_traj
                 ):
        #expects as list of items for each initalization variable
        self.state = np.array(state)
        self.action = np.array(action)
        self.action_mean = np.array(action_mean)
        self.reward = np.array(reward)
        self.next_state = np.array(next_state)
        self.initial_state = np.array(inital_state)
        self.done = np.array(done)
        self.time_step = np.array(time_step)
        self.optim_traj = np.array(optim_traj)


    def get_all_attributes(self):
        return [self.state, self.action, self.action_mean, self.reward, self.next_state, self.initial_state, self.done, self.time_step, self.optim_traj]

class Replay_Memory():

    def __init__(self, capacity=10000):
        self.no_data = 0
        self.position = 0
        self.capacity = capacity
        self.state, self.action, self.action_mean, self.reward, self.next_state, self.inital_state, self.done, self.time_step, self.optim_traj = [], [], [], [], [], [], [], [], []

    def push(self, state, action, action_mean, reward, next_state, inital_state, done, time_step, optim_traj):
        if len(self.state) < self.capacity:
            self.state.append(None)
            self.action.append(None)
            self.action_mean.append(None)
            self.reward.append(None)
            self.next_state.append(None)
            self.inital_state.append(None)
            self.done.append(None)
            self.time_step.append(None)
            self.optim_traj.append(None)
            self.no_data += 1


        self.state[self.position] = state
        self.action[self.position] = action
        self.action_mean[self.position] = action_mean
        self.reward[self.position] = reward
        self.next_state[self.position] = next_state
        self.inital_state[self.position] = inital_state
        self.done[self.position] = done
        self.time_step[self.position] = time_step
        self.optim_traj[self.position] = optim_traj

        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):

        if len(self.state) < self.capacity:
            indices = np.random.choice(len(self.state), batch_size)
        else:
            indices = np.random.choice(self.capacity, batch_size)


        state = np.take(np.array(self.state), indices, axis=0)
        action = np.take(np.array(self.action), indices, axis=0)
        action_mean = np.take(np.array(self.action_mean), indices, axis=0)
        reward = np.take(np.array(self.reward), indices, axis=0)
        next_state = np.take(np.array(self.next_state), indices, axis=0)
        initial_state = np.take(np.array(self.inital_state), indices, axis=0)
        done = np.take(np.array(self.done), indices, axis=0)
        time_step = np.take(np.array(self.time_step), indices, axis=0)
        optim_traj = np.take(np.array(self.optim_traj), indices, axis=0)


        return Transition_tuple(state, action, action_mean, reward, next_state, initial_state, done, time_step, optim_traj)

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
    action_mean = np.array([])
    reward = np.array([])
    next_state = np.array([])
    initial_state = np.array([])
    done = np.array([])
    time_step = np.array([])
    optim_traj = np.array([])

    i = 0
    for tuple in tuples:

        s, a, a_m, r, n_s, i_s, d, t, o_t = tuple.state, tuple.action, tuple.action_mean, tuple.reward, tuple.next_state, tuple.initial_state, tuple.done, tuple.time_step, tuple.optim_traj
        if i == 0:


            state = s
            action = a
            action_mean = a_m
            reward = r
            next_state = n_s
            initial_state = i_s
            done = d
            time_step = t
            optim_traj = o_t


        else:

            #print(np.shape(next_state[0]), np.shape(n_s[0]), np.shape(s[0]))
            #print(np.shape(next_state), np.shape(n_s), np.shape(s), np.shape(state))

            #print(next_state)

            state = np.concatenate((state, s), axis=0)
            action = np.concatenate((action, a), axis=0)
            action_mean = np.concatenate((action_mean, a_m), axis=0)
            reward = np.concatenate((reward, r), axis=0)
            #next_state = np.concatenate((next_state, n_s), axis=0)
            initial_state = np.concatenate((initial_state, i_s), axis=0)
            time_step = np.concatenate((done, d), axis=0)
            time_step = np.concatenate((time_step, t), axis=0)
            optim_traj = np.concatenate((optim_traj, o_t), axis=0)

        i += 1

    main_tuple = Transition_tuple(state, action, action_mean, reward, next_state, initial_state, done,  time_step, optim_traj)
    return main_tuple


