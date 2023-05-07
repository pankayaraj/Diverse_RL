import torch
from model import Discrete_Q_Function_NN
from parameters import NN_Paramters, Algo_Param, Save_Paths, Load_Paths
import numpy as np
from epsilon_greedy import epsilon_greedy
import math
import numpy as np
from log_ratio_1 import Log_Ratio
from ratio_1 import Ratio

from memory import Replay_Memory, combine_transition_tuples
from epsilon_greedy import epsilon_greedy
from util.q_learning_to_policy import Q_learner_Policy
class DivQL():

    def __init__(self, env, inital_log_buffer, q_nn_param, nu_pram, algo_param,  max_episodes =100, memory_capacity =10000,
                 log_ratio_memory_capacity=1000,
                 batch_size=400, num_z=2, save_path = Save_Paths(), load_path= Load_Paths(), ):

        self.num_z = num_z

        self.state_dim = q_nn_param.state_dim
        self.action_dim = q_nn_param.action_dim
        self.q_nn_param = q_nn_param
        self.algo_param = algo_param
        self.max_episodes = max_episodes

        self.save_path = Save_Paths()
        self.load_path = Load_Paths()

        self.inital_state = None
        self.time_step = 0

        self.max_episode_length = max_episodes

        self.Q = {}
        self.Target_Q = {}
        self.Q_optim = {}
        for i in range(self.num_z):
            self.Q[i] = Discrete_Q_Function_NN(nn_params=q_nn_param,
                                            save_path= self.save_path.q_path, load_path=self.load_path.q_path, )

            self.Target_Q[i] = Discrete_Q_Function_NN(nn_params=q_nn_param,
                                            save_path= self.save_path.q_path, load_path=self.load_path.q_path,  )
            self.Target_Q[i].load_state_dict(self.Q[i].state_dict())
            self.Q_optim[i] = torch.optim.Adam(self.Q[i].parameters(), self.q_nn_param.l_r)




        #self.loss_function = torch.nn.functional.smooth_l1_loss
        self.loss_function = torch.nn.functional.mse_loss


        self.memory = Replay_Memory(memory_capacity)
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.env = env

        # log ratio
        self.log_ratio_memory = {}
        self.log_ratio_memory_capacity = log_ratio_memory_capacity
        self.initalzie_replay_buffer(inital_log_buffer)
        self.current_index = 0

        self.occupancy_measure_prev = self.set_occupancy_measure_prev(inital_log_buffer)



    def initalzie_replay_buffer(self, log_ratio_memory_inital):
        self.log_ratio_memory[0] = log_ratio_memory_inital



    def add_new_replay_buffer(self, current_z):

        log_ratio_memory = {}
        #since the indexing starts from 0 we need z+1 to count the new buffer
        for i in range(current_z+1):
            log_ratio_memory[i] = Replay_Memory(self.log_ratio_memory_capacity)

        for i, (k, v) in enumerate(self.log_ratio_memory.items()):
            log_ratio_memory[k] = v

        self.log_ratio_memory = log_ratio_memory
        self.memory = Replay_Memory(self.memory_capacity)
        self.current_index = current_z

    def set_occupancy_measure_prev(self, inital_log_buffer):
        self.occupancy_measure_prev = [[0 for i in range(10)] for j in range(10)]
        for i in range(10):
            batch = inital_log_buffer.sample(self.batch_size)
            state = batch.state
            for j in range(self.batch_size):
                self.occupancy_measure_prev[state[j][1]][state[j][0]] += 1

        self.prev_O_M = np.array(self.occupancy_measure_prev) / np.sum(self.occupancy_measure_prev)
        return self.prev_O_M

    def reset_occupancy_measure_prev(self):
        #self.occupancy_measure_prev = [[0 for i in range(10)] for j in range(10)]


        for i in range(10):
            batch = self.log_ratio_memory[self.current_index].sample(self.batch_size)
            state = batch.state
            for j in range(self.batch_size):

                self.occupancy_measure_prev[state[j][1]][state[j][0]] += 1

        self.prev_O_M = np.array(self.occupancy_measure_prev) / np.sum(self.occupancy_measure_prev)
        return self.prev_O_M

    def save_main(self, q_path, target_q_path):
        torch.save(self.Q, q_path)
        torch.save(self.Target_Q, target_q_path)

    def load_main(self, q_path, target_q_path):
        self.Q = torch.load(q_path)
        self.Target_Q = torch.load( target_q_path)
    def save(self, z, q_path, target_q_path, nu_path1, nu_path2):
        self.Q[z].save(q_path)
        self.Target_Q[z].save(target_q_path)
        #self.log_ratio[z].nu_network.save(nu_path1)
        #self.ratio[z].nu_network.save(nu_path2)

    def load(self, z, q_path, target_q_path, nu_path1, nu_path2):
        self.Q[z].load(q_path)
        self.Target_Q[z].load(target_q_path)
        #self.log_ratio[z].nu_network.load(nu_path1)
        #self.ratio[z].nu_network.load(nu_path1)


    def change_z(self, current_z):

        self.add_new_replay_buffer(current_z)

        for t in range(500):
            state = self.env.reset()
            inital_state = state
            time_step = 0
            R = 0

            z = current_z-1

            tuples = []
            for i in range(self.max_episode_length):
                q_values = self.Q[z].get_value(state, format="numpy")
                action_scaler = np.argmax(q_values)
                next_state, reward, done, _ = self.env.step(action_scaler)

                R += reward
                # converting the action for buffer as one hot vector
                sample_hot_vec = np.array([0.0 for i in range(self.q_nn_param.action_dim)])
                sample_hot_vec[action_scaler] = 1
                action = sample_hot_vec
                time_step += 1

                if done:
                    tuples.append([state, action, reward, next_state, self.inital_state, self.time_step])
                    state = self.env.reset()
                    self.inital_state = state
                    self.time_step = 0
                    break

                if self.time_step == self.max_episode_length - 1:
                    tuples.append([state, action, reward, next_state, self.inital_state, self.time_step])
                    state = self.env.reset()
                    self.inital_state = state
                    self.time_step = 0
                    break

                tuples.append([state, action, reward, next_state, self.inital_state, self.time_step])
                state = next_state

            for i in range(len(tuples)):
                tuples[i].append(True)

            for t in tuples:
                self.log_ratio_memory[z+1].push(t[0], t[1], t[2], t[3], t[4], t[5], t[6])


        self.reset_occupancy_measure_prev()
    def step(self, state):

        #since step is done on the basis of single states and not as a batch
        batch_size = 1

        z = self.current_index


        q_values = self.Target_Q[z].get_value(state, format="numpy")
        action, self.steps_done, self.epsilon = epsilon_greedy(q_values, self.steps_done, self.epsilon, self.action_dim)

        next_state, reward, done, _ = self.env.step(action)

        #adding alternative reward
        reward += 1*0.01/(self.prev_O_M[next_state[1]][next_state[0]]+0.01)
        #reward += 1 * math.log(0.01 / (self.prev_O_M[next_state[1]][next_state[0]] + 0.01))

        #converting the action for buffer as one hot vector
        sample_hot_vec = np.array([0.0 for i in range(self.q_nn_param.action_dim)])
        sample_hot_vec[action] = 1
        action = sample_hot_vec

        self.time_step += 1

        if done:
            next_state = None
            self.memory.push(state, action, reward, next_state, self.inital_state, self.time_step, True)
            state = self.env.reset()
            self.inital_state = state
            self.time_step = 0
            return state

        if self.time_step == self.max_episodes:
            self.memory.push(state, action, reward, next_state, self.inital_state, self.time_step, True)
            state = self.env.reset()
            self.inital_state = state
            self.time_step = 0
            return state

        self.memory.push(state, action, reward, next_state, self.inital_state, self.time_step, True)
        return next_state


    def get_action(self, state, z):

        q_values = self.Q[z].get_value(state, format="numpy")
        action_scaler = np.argmax(q_values)
        return action_scaler

    def train(self):

        batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        state = batch.state
        action = torch.Tensor(batch.action)
        action_scaler = action.max(1)[1].unsqueeze(1) #to make them as indices in the gather function
        reward = torch.Tensor(batch.reward)
        next_state = batch.next_state

        z = self.current_index

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.q_nn_param.device, dtype=torch.bool)

        non_final_next_states = torch.Tensor([s for s in next_state if s is not None])

        #get only the q value relevant to the actions
        state_action_values = self.Q[z].get_value(state).gather(1, action_scaler)


        with torch.no_grad():
            next_state_action_values = torch.zeros(batch_size, device=self.q_nn_param.device)
            next_state_action_values[non_final_mask] = self.Target_Q[z].get_value(non_final_next_states).max(1)[0]
            #now there will be a zero if it is the final state and q*(n_s,n_a) is its not None

        expected_state_action_values = (self.algo_param.gamma*next_state_action_values).unsqueeze(1) + reward.unsqueeze(1)
        loss = self.loss_function( state_action_values, expected_state_action_values)

        self.Q_optim[z].zero_grad()
        loss.backward()
        self.Q_optim[z].step()

    def hard_update(self):
        self.Target_Q[self.current_index].load_state_dict(self.Q[self.current_index].state_dict())

    def initalize(self):

        #inital_phase train after this by continuing with step and train at single iteration and hard update at update interval
        self.steps_done = 0
        self.epsilon = 0.99
        state = self.env.reset()
        self.inital_state = state
        for i in range(self.batch_size):
            state = self.step(state)
        return state





