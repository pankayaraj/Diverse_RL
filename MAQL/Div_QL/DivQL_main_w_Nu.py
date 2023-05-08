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
        self.nu_param = nu_pram
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

        self.log_ratio = {}
        self.ratio = {}
        for i in range(self.num_z):
            self.log_ratio[i] = Log_Ratio(self.nu_param, self.algo_param,  num_z=self.num_z,
                                 save_path=save_path.nu_path, load_path=load_path.nu_path, state_action=False)
            self.ratio[i]  = Ratio(self.nu_param, self.algo_param,  num_z=self.num_z,
                                 save_path=save_path.nu_path, load_path=load_path.nu_path, state_action=False)




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

        self.ratio_batch_size = 1000




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



    def save_main(self, q_path, target_q_path):
        torch.save(self.Q, q_path)
        torch.save(self.Target_Q, target_q_path)

    def load_main(self, q_path, target_q_path):
        self.Q = torch.load(q_path)
        self.Target_Q = torch.load( target_q_path)



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




    def train_log_ratio(self, z, fixed_ratio_memory=False):
        #here data must be off the ratio's memory

        if fixed_ratio_memory:
            data1 = None
            data2 = None
        else:
            data2 = self.uniform_sampler(self.ratio_batch_size )

            data_temp = self.uniform_sampler(int(self.ratio_batch_size*0.0))
            data1 = self.sample_for_log_others(self.ratio_batch_size , current_z_index=z)


            for i in range(int(self.ratio_batch_size*0.0)):
                data1.state[i] = data_temp.state[i]
                data1.action[i] = data_temp.action[i]



        z_hot_vec = np.array([0.0 for i in range(self.num_z)])
        z_hot_vec[z] = 1
        z_arr = np.array([z_hot_vec for _ in range(self.ratio_batch_size )])

        self.log_ratio[z].train_ratio(data1, data2, z_arr )

    def train_ratio(self, z, fixed_ratio_memory=False):
        #here data must be off the ratio's memory

        if fixed_ratio_memory:
            data1 = None
            data2 = None
        else:
            data2 = self.uniform_sampler(self.ratio_batch_size )
            data_temp = self.uniform_sampler(int(self.ratio_batch_size * 0.0))
            data1 = self.sample_for_log_others(self.ratio_batch_size, current_z_index=z)

            for i in range(int(self.ratio_batch_size * 0.0)):
                data1.state[i] = data_temp.state[i]
                data1.action[i] = data_temp.action[i]

        z_hot_vec = np.array([0.0 for i in range(self.num_z)])
        z_hot_vec[z] = 1
        z_arr = np.array([z_hot_vec for _ in range(self.ratio_batch_size )])

        self.ratio[z].train_dice(data1, data2, z_arr)
    def step(self, state):

        #since step is done on the basis of single states and not as a batch
        batch_size = 1

        z = self.current_index


        q_values = self.Target_Q[z].get_value(state, format="numpy")
        action, self.steps_done, self.epsilon = epsilon_greedy(q_values, self.steps_done, self.epsilon, self.action_dim)

        next_state, reward, done, _ = self.env.step(action)

        #adding alternative reward

        D = Data()
        D.state = next_state
        D.action = action

        reward += 5*self.ratio[z].get_state_action_density_ratio(D, None).item()
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

    def train_ratios(self, no):

        z = self.current_index
        for _ in range(no):
            if _ %500 == 0:
                print(_)
                print(self.ratio[z].debug_V["x**2"])
                print(self.ratio[z].debug_V["linear"] )

                print(self.log_ratio[z].debug_V["log_exp"])
                print(self.log_ratio[z].debug_V["linear"])

            self.train_ratio(z)
            self.train_log_ratio(z)
    def train(self):
        z = self.current_index



        batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        state = batch.state
        action = torch.Tensor(batch.action)
        action_scaler = action.max(1)[1].unsqueeze(1) #to make them as indices in the gather function
        reward = torch.Tensor(batch.reward)
        next_state = batch.next_state



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



    def sample_for_log_others(self, batch_size, current_z_index):
        n = batch_size//(self.current_index+1)
        mem_indices = [i  for i in range(self.current_index+1) ]


        mem_sample_size = [n for i in range(self.current_index+1)]

        if n*(self.current_index) != batch_size:
            mem_sample_size[-1] += batch_size - (self.current_index+1)*(n)

        samples = []

        for i in range(self.current_index+1):

            samples.append(self.log_ratio_memory[mem_indices[i]].sample(mem_sample_size[i]))

        #if self.current_index == 2:
        #    print(samples[0].next_state, samples[1].next_state)
        data = combine_transition_tuples(samples)
        return data

    def uniform_sampler(self, batch_size):

        action_s = np.random.randint(4, size=batch_size)
        state = np.random.randint(10, size=(batch_size, self.state_dim))


        action = [[0.0 for i in range(self.action_dim)] for j in range(batch_size)]
        for i in range(len(action_s)):
            action[i][action_s[i]] = 1
        action = np.array(action)


        D = Data()
        D.state = state
        D.action = action

        return  D

class Data():
    def __init__(self):
        self.state = None
        self.action = None

