import torch
import numpy as np
from log_ratio import Log_Ratio
from ratio import Ratio

from model import Discrete_Q_Function_NN_Z
from parameters import NN_Paramters, Algo_Param, Save_Paths, Load_Paths
from memory import Replay_Memory, combine_transition_tuples
from epsilon_greedy import epsilon_greedy
from util.q_learning_to_policy import Q_learner_Policy


class DivQL():

    def __init__(self, env, q_nn_param, nu_param, algo_param, num_z, optim_alpha=0.8, max_episode_length =100, memory_capacity =10000,
                 log_ratio_memory_capacity=5000, batch_size=400, save_path = Save_Paths(), load_path= Load_Paths(),
                 deterministic_env=True, average_next_nu = True):



        self.state_dim = q_nn_param.state_dim
        self.action_dim = q_nn_param.action_dim
        self.q_nn_param = q_nn_param
        self.algo_param = algo_param
        self.nu_param = nu_param
        self.num_z = num_z

        self.max_episode_length = max_episode_length

        self.save_path = Save_Paths()
        self.load_path = Load_Paths()

        self.inital_state = None
        self.time_step = 0

        self.Q = Discrete_Q_Function_NN_Z(nn_params=q_nn_param,
                                        save_path= self.save_path.q_path, load_path=self.load_path.q_path, num_z=num_z)

        self.Target_Q = Discrete_Q_Function_NN_Z(nn_params=q_nn_param,
                                        save_path= self.save_path.q_path, load_path=self.load_path.q_path,  num_z=num_z)
        self.Target_Q.load_state_dict(self.Q.state_dict())

        #self.loss_function = torch.nn.functional.smooth_l1_loss
        self.loss_function = torch.nn.functional.mse_loss
        self.Q_optim = torch.optim.Adam(self.Q.parameters(), self.q_nn_param.l_r)

        self.log_ratio = Log_Ratio(self.nu_param, self.algo_param,  num_z=self.num_z,
                             save_path=save_path.nu_path, load_path=load_path.nu_path)
        self.ratio  = Ratio(self.nu_param, self.algo_param,  num_z=self.num_z,
                             save_path=save_path.nu_path, load_path=load_path.nu_path)

        #q learning
        self.memory = {}
        for i in range(num_z):
            self.memory[i] = Replay_Memory(memory_capacity)
        self.memory_capacity = memory_capacity

        #log ratio
        self.log_ratio_memory = {}
        for i in range(num_z):
            self.log_ratio_memory[i] = Replay_Memory(log_ratio_memory_capacity)
        self.log_ratio_memory_capacity = log_ratio_memory_capacity

        self.batch_size = batch_size
        self.env = env

        self.L = 0
        self.T = 0


    def train_log_ratio(self, z, fixed_ratio_memory=False):
        #here data must be off the ratio's memory

        if fixed_ratio_memory:
            data1 = None
            data2 = None
        else:
            data1 = self.sample_for_ratio_current(self.batch_size, current_z_index=z)
            data2 = self.sample_for_log_others(self.batch_size, current_z_index=z)

        z_hot_vec = np.array([0.0 for i in range(self.num_z)])
        z_hot_vec[z] = 1
        z_arr = np.array([z_hot_vec for _ in range(self.batch_size)])

        self.log_ratio.train_ratio(data1, data2, z_arr )

    def train_ratio(self, z, fixed_ratio_memory=False):
        #here data must be off the ratio's memory

        if fixed_ratio_memory:
            data1 = None
            data2 = None
        else:
            data1 = self.sample_for_ratio_current(self.batch_size, current_z_index=z)
            data2 = self.sample_for_log_others(self.batch_size, current_z_index=z)

        z_hot_vec = np.array([0.0 for i in range(self.num_z)])
        z_hot_vec[z] = 1
        z_arr = np.array([z_hot_vec for _ in range(self.batch_size)])

        self.ratio.train_dice(data1, data2, z_arr)
    def get_log_ratio_others(self, data, z_arr):
        #here data can be off Q learning's memory
        target_policy = self.get_target_policy(target=True)
        return self.log_ratio.get_log_state_action_density_ratio(data, z_arr, target_policy)

    #this function is in case we need to data externally for the log ratio training memory
    def push_ratio_memory(self, current_index, state, action, reward, next_state, inital_state, time_step, optim_traj):
        self.log_ratio_memory[current_index].push(state, action, reward, next_state, inital_state, time_step, optim_traj)

    def step(self, R_max):

        #since step is done on the basis of single states and not as a batch
        batch_size = 1

        self.steps_done = 0
        self.epsilon = 0.9
        state = self.env.reset()
        self.inital_state = state

        tuples = []
        R = 0

        z = np.random.randint(0, self.num_z)
        # converting z into one hot vector
        z_hot_vec = np.array([0.0 for i in range(self.num_z)])
        z_hot_vec[z] = 1

        for i in range(self.max_episode_length):
            self.T += 1
            q_values = self.Target_Q.get_value(state, z_hot_vec, format="numpy")
            action, self.steps_done, self.epsilon = epsilon_greedy(q_values, self.steps_done, self.epsilon, self.action_dim)

            next_state, reward, done, _ = self.env.step(action)

            R += reward
            #converting the action for buffer as one hot vector
            sample_hot_vec = np.array([0.0 for i in range(self.q_nn_param.action_dim)])
            sample_hot_vec[action] = 1
            action = sample_hot_vec

            self.time_step += 1

            if done:
                next_state = None
                tuples.append([state, action, reward, next_state, self.inital_state, self.time_step])
                state = self.env.reset()
                self.inital_state = state
                self.time_step = 0

                break

            if self.time_step == self.max_episode_length-1:
                tuples.append([state, action, reward, next_state, self.inital_state, self.time_step])
                state = self.env.reset()
                self.inital_state = state
                self.time_step = 0

                break

            tuples.append([state, action, reward, next_state, self.inital_state, self.time_step])

            state = next_state

            if self.T%100 == 0:
                self.eval(20, R_max)

            if self.T%1000 == 0:
                self.hard_update()
            if self.T > 1000:

                self.train(z, True)
                self.train_log_ratio(z)
                self.train_ratio(z)

        if R_max < 0.8*R:
            for i in range(len(tuples)):
                tuples[i].append(True)
        else:
            for i in range(len(tuples)):
                tuples[i].append(False)
        for t in tuples:
            self.memory[z].push(t[0], t[1], t[2], t[3], t[4], t[5], t[6])



    def eval(self, eval_steps, R_max):

        #this function is to push into the memory of teh evaluation buffer for ratio computation
        for t in range(eval_steps):
            state = self.env.reset()
            inital_state = state
            time_step = 0
            R = 0

            z = np.random.randint(0, self.num_z)
            # converting z into one hot vector
            z_hot_vec = np.array([0.0 for i in range(self.num_z)])
            z_hot_vec[z] = 1
            tuples = []
            for i in range(self.max_episode_length):
                q_values = self.Target_Q.get_value(state, z_hot_vec, format="numpy")
                action_scaler = np.argmax(q_values)
                next_state, reward, done, _ = self.env.step(action_scaler)
                R += reward
                # converting the action for buffer as one hot vector
                sample_hot_vec = np.array([0.0 for i in range(self.q_nn_param.action_dim)])
                sample_hot_vec[action_scaler] = 1
                action = sample_hot_vec

                time_step += 1

                if done:
                    next_state = None
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

            if R_max < 0.8 * R:
                for i in range(len(tuples)):
                    tuples[i].append(True)
            else:
                for i in range(len(tuples)):
                    tuples[i].append(False)
            for t in tuples:
                self.log_ratio_memory[z].push(t[0], t[1], t[2], t[3], t[4], t[5], t[6])


    def train(self, z, log_ratio_update=False):

        batch_size = self.batch_size
        if len(self.memory[z]) < batch_size:
            return

        batch = self.memory[z].sample(batch_size)
        state = batch.state
        action = torch.Tensor(batch.action).to(self.q_nn_param.device)
        action_scaler = action.max(1)[1].unsqueeze(1) #to make them as indices in the gather function
        reward = torch.Tensor(batch.reward).to(self.q_nn_param.device)
        next_state = batch.next_state
        optim_traj = batch.optim_traj

        # converting z into one hot vector
        z_hot_vec = np.array([0.0 for i in range(self.num_z)])
        z_hot_vec[z] = 1


        z_arr = np.array([z_hot_vec for _ in range(batch_size)])
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.q_nn_param.device, dtype=torch.bool)
        non_final_next_states = torch.Tensor([s for s in next_state if s is not None]).to(self.q_nn_param.device)
        optim_mask = torch.Tensor(optim_traj).bool()

        #get only the q value relevant to the actions
        state_action_values = self.Q.get_value(state, z_arr).gather(1, action_scaler)
        with torch.no_grad():
            next_state_action_values = torch.zeros(batch_size, device=self.q_nn_param.device)
            next_state_action_values[non_final_mask] = self.Target_Q.get_value(non_final_next_states, z_arr).max(1)[0]
            #now there will be a zero if it is the final state and q*(n_s,n_a) is its not None

      
        with torch.no_grad():
            log_ratio = self.get_log_ratio(batch, z_arr)
            ratio = self.get_ratio(batch, z_arr)

            #since log ratio is maximised only for optimal trajectories
            effective_log_ratio = log_ratio.squeeze()*optim_mask
            effective_ratio = ratio.squeeze()*optim_mask


        expected_state_action_values = (self.algo_param.gamma*next_state_action_values).unsqueeze(1) + reward.unsqueeze(1) + self.algo_param.alpha*effective_log_ratio.unsqueeze(1)
        loss = self.loss_function( state_action_values, expected_state_action_values)

        self.Q_optim.zero_grad()
        loss.backward()
        self.Q_optim.step()




    def hard_update(self):
        self.Target_Q.load_state_dict(self.Q.state_dict())

    def initalize(self):

        #inital_phase train after this by continuing with step and train at single iteration and hard update at update interval
        self.steps_done = 0
        self.epsilon = 0.9
        state = self.env.reset()
        self.inital_state = state
        for i in range(self.batch_size):
            state = self.step(state)
        return state

    def sample_for_log_others(self, batch_size, current_z_index):
        n = batch_size//(self.num_z-1)
        mem_indices = [i and i != current_z_index for i in range(self.num_z)]
        mem_sample_size = [n for i in range(self.num_z-1)]
        if n*(self.num_z-1) != batch_size:
            mem_sample_size[-1] += batch_size - (self.num_z-1)*(n)
        samples = []
        for i in range(self.num_z-1):
            samples.append(self.memory[mem_indices[i]].sample(mem_sample_size[i]))

        data = combine_transition_tuples(samples)
        return data

    def sample_for_ratio_current(self, batch_size, current_z_index):
        data = self.log_ratio_memory[current_z_index].sample(batch_size)
        return data

    def sample_for_Q(self, batch_size, current_z_index):
        data = self.memory[current_z_index].sample(batch_size)
        return data

    def get_target_policy(self, target=True):
        #this is the current policy the agent should evaluate against given the data
        #choose weather to give the current Q or the target Q from dual_Q
        if target:
            target_policy = Q_learner_Policy(self.Target_Q, self.q_nn_param)
        else:
            target_policy = Q_learner_Policy(self.Q, self.q_nn_param)
        return target_policy


    def get_action(self, state, z):
        # converting z into one hot vector
        z_hot_vec = np.array([0.0 for i in range(self.num_z)])
        z_hot_vec[z] = 1
        q_values = self.Q.get_value(state, z_hot_vec, format="numpy")
        action_scaler = np.argmax(q_values)
        return action_scaler

    def get_log_ratio(self, data, z_arr):
        return self.log_ratio.get_log_state_action_density_ratio(data, z_arr)

    def get_ratio(self, data, z_arr):
        return self.ratio.get_state_action_density_ratio(data, z_arr)
    def save(self, q_path, target_q_path, nu_path1, nu_path2):
        self.Q.save(q_path)
        self.Target_Q.save(target_q_path)
        self.log_ratio.nu_network.save(nu_path1)
        self.ratio.nu_network.save(nu_path2)

    def load(self, q_path, target_q_path, nu_path1, nu_path2):
        self.Q.load(q_path)
        self.Target_Q.load(target_q_path)
        self.log_ratio.nu_network.load(nu_path1)
        self.ratio.nu_network.load(nu_path1)

