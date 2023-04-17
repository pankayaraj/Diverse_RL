import torch
from model import Discrete_Q_Function_NN, SF_NN
from parameters import NN_Paramters, Algo_Param, Save_Paths, Load_Paths
from memory import Replay_Memory
import numpy as np
from epsilon_greedy import epsilon_greedy


class Q_learning():

    def __init__(self, env, q_nn_param, algo_param, max_episodes =100, memory_capacity =10000,
                 batch_size=400, num_z=2, save_path = Save_Paths(), load_path= Load_Paths()):

        self.state_dim = q_nn_param.state_dim
        self.action_dim = q_nn_param.action_dim
        self.q_nn_param = q_nn_param
        self.algo_param = algo_param
        self.max_episodes = max_episodes

        self.save_path = Save_Paths()
        self.load_path = Load_Paths()

        self.inital_state = None
        self.time_step = 0
        self.num_z = 2


        self.Q = Discrete_Q_Function_NN(nn_params=q_nn_param,
                                        save_path= self.save_path.q_path, load_path=self.load_path.q_path,)

        self.Target_Q = Discrete_Q_Function_NN(nn_params=q_nn_param,
                                        save_path= self.save_path.q_path, load_path=self.load_path.q_path,)
        self.Target_Q.load_state_dict(self.Q.state_dict())

        #self.loss_function = torch.nn.functional.smooth_l1_loss
        self.loss_function = torch.nn.functional.mse_loss
        self.Q_optim = torch.optim.Adam(self.Q.parameters(), self.q_nn_param.l_r)

        self.SF = SF_NN(nn_params=q_nn_param,
                           save_path=self.save_path.q_path, load_path=self.load_path.q_path, )

        self.Target_SF = SF_NN(nn_params=q_nn_param,
                                  save_path=self.save_path.q_path, load_path=self.load_path.q_path, )

        self.Target_SF.load_state_dict(self.SF.state_dict())
        self.SF_optim = torch.optim.Adam(self.SF.parameters(), self.q_nn_param.l_r)


        self.memory = Replay_Memory(memory_capacity)
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.env = env



    def save(self, q_path, target_q_path, sf_path, target_sf_path):
        self.Q.save(q_path)
        self.Target_Q.save(target_q_path)
        self.SF.save(sf_path)
        self.Target_SF.save(target_sf_path)

    def load(self, q_path, target_q_path, sf_path, target_sf_path):
        self.Q.load(q_path)
        self.Target_Q.load(target_q_path)
        self.SF.load(sf_path)
        self.Target_SF.load(target_sf_path)

    def step(self, state):

        #since step is done on the basis of single states and not as a batch
        batch_size = 1

        z = 0
        # converting z into one hot vector
        z_hot_vec = np.array([0.0 for i in range(self.num_z)])
        z_hot_vec[z] = 1


        q_values = self.Target_Q.get_value(state, format="numpy")
        action, self.steps_done, self.epsilon = epsilon_greedy(q_values, self.steps_done, self.epsilon, self.action_dim)

        next_state, reward, done, _ = self.env.step(action)

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

    def get_action(self, state):

        z = 0
        z_hot_vec = np.array([0.0 for i in range(self.num_z)])
        z_hot_vec[z] = 1

        q_values = self.Q.get_value(state, format="numpy")
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

        z = 0
        z_hot_vec = np.array([0.0 for i in range(self.num_z)])
        z_hot_vec[z] = 1
        z_arr = np.array([z_hot_vec for _ in range(batch_size)])



        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.q_nn_param.device, dtype=torch.bool)

        non_final_next_states = torch.Tensor([s for s in next_state if s is not None])


        #get only the q value relevant to the actions
        state_action_values = self.Q.get_value(state).gather(1, action_scaler)
        SF_values = self.SF.get_value(state, action)

        with torch.no_grad():
            next_state_action_values = torch.zeros(batch_size, device=self.q_nn_param.device)
            next_state_action_values[non_final_mask] = self.Target_Q.get_value(non_final_next_states).max(1)[0]
            #now there will be a zero if it is the final state and q*(n_s,n_a) is its not None

            next_q_values = self.Q.get_value(non_final_next_states)
            max = torch.argmax(next_q_values, dim=1)
            non_final_next_action = torch.zeros(next_q_values.shape)
            non_final_next_action.scatter_(1, max.unsqueeze(1), 1)

            next_SF_values = self.SF.get_value(non_final_next_states, non_final_next_action)

        expected_SF_values = (self.algo_param.gamma * next_SF_values) + state  # here state itself is treated as sucessor feture /phi
        loss_SF = self.loss_function(SF_values, expected_SF_values.float())

        self.SF_optim.zero_grad()
        loss_SF.backward()
        self.SF_optim.step()


        expected_state_action_values = (self.algo_param.gamma*next_state_action_values).unsqueeze(1) + reward.unsqueeze(1)
        loss = self.loss_function( state_action_values, expected_state_action_values)
        self.Q_optim.zero_grad()
        loss.backward()
        self.Q_optim.step()

    def hard_update(self):
        self.Target_Q.load_state_dict(self.Q.state_dict())
        self.Target_SF.load_state_dict(self.SF.state_dict())
    def initalize(self):

        #inital_phase train after this by continuing with step and train at single iteration and hard update at update interval
        self.steps_done = 0
        self.epsilon = 0.99
        state = self.env.reset()
        self.inital_state = state
        for i in range(self.batch_size):
            state = self.step(state)
        return state





