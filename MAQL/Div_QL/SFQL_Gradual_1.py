import torch
import numpy as np
from log_ratio_1 import Log_Ratio
from ratio_1 import Ratio

from model import Discrete_Q_Function_NN
from parameters import NN_Paramters, Algo_Param, Save_Paths, Load_Paths
from memory import Replay_Memory, combine_transition_tuples
from epsilon_greedy import epsilon_greedy
from util.q_learning_to_policy import Q_learner_Policy


class DivQL_Gradual():

    def __init__(self, env,  inital_log_buffer, q_nn_param, nu_param, algo_param, num_z, optim_alpha=0.8, max_episode_length =100, memory_capacity =10000,
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

        self.SF = {}
        self.Target_SF = {}
        self.SF_optim = {}


        for i in range(self.num_z):
            self.SF[i] = Discrete_Q_Function_NN(nn_params=q_nn_param,
                                            save_path= self.save_path.q_path, load_path=self.load_path.q_path, )

            self.Target_SF[i] = Discrete_Q_Function_NN(nn_params=q_nn_param,
                                            save_path= self.save_path.q_path, load_path=self.load_path.q_path, )

            self.Target_SF[i].load_state_dict(self.SF[i].state_dict())
            self.SF_optim[i] = torch.optim.Adam(self.SF[i].parameters(), self.q_nn_param.l_r)


        self.memory = Replay_Memory(memory_capacity)
        self.memory_capacity = memory_capacity



        self.batch_size = batch_size
        self.env = env

        self.L = 0
        self.T = 0

        self.current_index = 0




    def step(self, R_max, current_z):

        #since step is done on the basis of single states and not as a batch
        batch_size = 1

        self.steps_done = 0
        self.epsilon = 0.9
        state = self.env.reset()
        self.inital_state = state

        tuples = []
        R = 0

        z = current_z

        for i in range(self.max_episode_length):
            self.T += 1
            q_values = self.Target_Q[z].get_value(state, format="numpy")
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


            if self.T%1000 == 0:
                self.hard_update(z)
            if self.T > 10000:

                self.train(z, True)


        if R_max < 0.8*R:
            for i in range(len(tuples)):
                tuples[i].append(True)
        else:
            for i in range(len(tuples)):
                tuples[i].append(False)
        for t in tuples:
            self.memory.push(t[0], t[1], t[2], t[3], t[4], t[5], t[6])




    def train(self, z, log_ratio_update=False):

        batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        state = batch.state
        action = torch.Tensor(batch.action).to(self.q_nn_param.device)
        action_scaler = action.max(1)[1].unsqueeze(1) #to make them as indices in the gather function
        reward = torch.Tensor(batch.reward).to(self.q_nn_param.device)
        next_state = batch.next_state
        optim_traj = batch.optim_traj



        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.q_nn_param.device, dtype=torch.bool)
        non_final_next_states = torch.Tensor([s for s in next_state if s is not None]).to(self.q_nn_param.device)
        optim_mask = torch.Tensor(optim_traj).bool()


        #get only the q value relevant to the actions
        state_action_values = self.Q[z].get_value(state).gather(1, action_scaler)
        with torch.no_grad():
            next_state_action_values = torch.zeros(batch_size, device=self.q_nn_param.device)
            next_state_action_values[non_final_mask] = self.Target_Q[z].get_value(non_final_next_states).max(1)[0]
            #now there will be a zero if it is the final state and q*(n_s,n_a) is its not None

      
        with torch.no_grad():
            log_ratio = self.get_log_ratio(batch, z_arr, z)
            ratio = self.get_ratio(batch, z_arr, z)

            #since log ratio is maximised only for optimal trajectories
            effective_log_ratio = log_ratio.squeeze()*optim_mask
            effective_ratio = ratio.squeeze()*optim_mask


        expected_state_action_values = (self.algo_param.gamma*next_state_action_values).unsqueeze(1) + reward.unsqueeze(1) + self.algo_param.alpha*effective_log_ratio.unsqueeze(1)
        loss = self.loss_function( state_action_values, expected_state_action_values)

        self.Q_optim[z].zero_grad()
        loss.backward()
        self.Q_optim[z].step()




    def hard_update(self, z):
        self.Target_Q[z].load_state_dict(self.Q[z].state_dict())
        self.Target_SF[z].load_state_dict(self.SF[z].state_dict())

    def initalize(self, R_max):

        #inital_phase train after this by continuing with step and train at single iteration and hard update at update interval
        self.steps_done = 0
        self.epsilon = 0.9
        self.T = 0
        state = self.env.reset()
        self.inital_state = state


        return state



    def sample_for_Q(self, batch_size, current_z_index):
        data = self.memory.sample(batch_size)
        return data

    def sample_for_SF(self, batch_size, current_z_index):
        data = self.memory.sample(batch_size)
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
        q_values = self.Q[z].get_value(state, format="numpy")
        action_scaler = np.argmax(q_values)
        return action_scaler


    def save(self, z, q_path, target_q_path, sf, target_sf):
        self.Q[z].save(q_path)
        self.Target_Q[z].save(target_q_path)
        self.SF[z].save(sf)
        self.Target_SF[z].save(target_sf)

    def load(self, z, q_path, target_q_path,  sf, target_sf):
        self.Q[z].load(q_path)
        self.Target_Q[z].load(target_q_path)
        self.SF[z].load(sf)
        self.Target_SF[z].load(target_sf)

