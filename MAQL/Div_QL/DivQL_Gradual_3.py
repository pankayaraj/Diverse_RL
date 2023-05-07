import torch
import numpy as np
from log_ratio_1 import Log_Ratio
from ratio_1 import Ratio

from model import Discrete_Q_Function_NN
from parameters import NN_Paramters, Algo_Param, Save_Paths, Load_Paths
#from memory import Replay_Memory, combine_transition_tuples
from memory_pq import Replay_Memory, combine_transition_tuples
from epsilon_greedy import epsilon_greedy
from util.q_learning_to_policy import Q_learner_Policy


class DivQL_Gradual():

    def __init__(self, env,  inital_log_buffer, q_nn_param, nu_param, algo_param, num_z, optim_alpha=0.8, max_episode_length =100, memory_capacity =2000,
                 log_ratio_memory_capacity=1000, batch_size=400, save_path = Save_Paths(), load_path= Load_Paths(),
                 deterministic_env=True, average_next_nu = True):

        #self.distance = "KL"
        #self.distance = "Jeffrey"
        #self.distance = "Pearson"
        #self.distance = "Exponential"
        self.distance = "A"

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


        #q learning

        self.memory = Replay_Memory(memory_capacity)
        self.memory_capacity = memory_capacity



        self.batch_size = batch_size
        self.env = env

        self.L = 0
        self.T = 0

        self.current_index = 0

        self.occupancy_measure = [[0 for i in range(10)] for j in range(10)]
        self.occupancy_measure_prev = self.set_occupancy_measure_prev(inital_log_buffer)


        self.instances = 0
    #zeroth case

    def reset_occupancy(self, ):
        self.occupancy_measure = [[0 for i in range(10)] for j in range(10)]



    def set_occupancy_measure_prev(self, inital_log_buffer):
        self.occupancy_measure_prev = [[0 for i in range(10)] for j in range(10)]
        for i in range(10):
            batch = inital_log_buffer.sample(self.batch_size)
            state = batch.state
            for j in range(self.batch_size):
                self.occupancy_measure_prev[state[j][1]][state[j][0]] += 1

        self.prev_O_M = np.array(self.occupancy_measure_prev) / np.sum(self.occupancy_measure_prev)
        return self.prev_O_M



    def step(self, R_max, current_z):

        #since step is done on the basis of single states and not as a batch
        batch_size = 1

        self.steps_done = 0
        #self.epsilon = 0.9
        state = self.env.reset()
        self.inital_state = state

        tuples = []
        R = 0

        z = current_z
        # converting z into one hot vector
        z_hot_vec = np.array([0.0 for i in range(self.num_z)])
        z_hot_vec[z] = 1

        for i in range(self.max_episode_length):
            self.T += 1
            q_values = self.Target_Q[z].get_value(state, format="numpy")
            action, self.steps_done, self.epsilon = epsilon_greedy(q_values, self.steps_done, self.epsilon, self.action_dim)

            next_state, reward, done, _ = self.env.step(action)

            p1 = 1 / 100
            p2 = self.prev_O_M[next_state[1]][next_state[0]]

            reward += (p1) / (p2 + 0.01)

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
                self.eval(20, R_max, z)

            if self.T%1000 == 0:
                self.hard_update(z)
            if self.T > 10000:

                self.train(z, True)

        priority = 0
        if 0.8*R_max < R and R<R_max:

            priority = 2
            self.instances += 1
            #else:
            #    priority = 0
            for i in range(len(tuples)):
                tuples[i].append(True)
        else:
            for i in range(len(tuples)):
                tuples[i].append(False)
        for t in tuples:
            self.memory.push(t[0], t[1], t[2], t[3], t[4], t[5], t[6], priority)



    def eval(self, eval_steps, R_max, current_z):

        self.reset_occupancy()

        #this function is to push into the memory of teh evaluation buffer for ratio computation
        for t in range(eval_steps):
            state = self.env.reset()
            inital_state = state
            time_step = 0
            R = 0

            z = current_z

            tuples = []
            for i in range(self.max_episode_length):
                q_values = self.Target_Q[z].get_value(state, format="numpy")
                action_scaler = np.argmax(q_values)


                self.occupancy_measure[state[1]][state[0]] += 1


                next_state, reward, done, _ = self.env.step(action_scaler)
                R += reward
                # converting the action for buffer as one hot vector
                sample_hot_vec = np.array([0.0 for i in range(self.q_nn_param.action_dim)])
                sample_hot_vec[action_scaler] = 1
                action = sample_hot_vec

                time_step += 1

                if done:
                    self.occupancy_measure[next_state[1]][next_state[0]] += 1
                    """
                    next_state = None
                    tuples.append([state, action, reward, next_state, self.inital_state, self.time_step])
                    state = self.env.reset()
                    self.inital_state = state
                    self.time_step = 0
                    """
                    break

                if self.time_step == self.max_episode_length - 1:
                    tuples.append([state, action, reward, next_state, self.inital_state, self.time_step])
                    state = self.env.reset()
                    self.inital_state = state
                    self.time_step = 0

                    break

                tuples.append([state, action, reward, next_state, self.inital_state, self.time_step])

                state = next_state

            if 0.8 * R_max <  R:

                for i in range(len(tuples)):
                    tuples[i].append(True)
            else:
                for i in range(len(tuples)):
                    tuples[i].append(False)



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

        # converting z into one hot vector
        z_hot_vec = np.array([0.0 for i in range(self.num_z)])
        z_hot_vec[z] = 1
        z_arr = np.array([z_hot_vec for _ in range(batch_size)])

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



        O_M = np.array(self.occupancy_measure)/np.sum(self.occupancy_measure)






        with torch.no_grad():
            ratio = []
            delta = 0.01
            for i in range(batch_size):

                if next_state[i] is not None:
                #p1 = O_M[state[i][1]][state[i][0]]
                    p1 = 1/100
                    p2 = self.prev_O_M[next_state[i][1]][next_state[i][0]]

                    ratio.append((p1)/(p2+0.01))
                else:
                    ratio.append(0)


            ratio =  torch.Tensor(ratio)
            log_ratio = torch.log(ratio)
            #since log ratio is maximised only for optimal trajectories
            effective_log_ratio = log_ratio.squeeze()*optim_mask
            effective_ratio = ratio.squeeze()*optim_mask

        self.dist = effective_log_ratio.unsqueeze(1)

        if self.distance == "KL":
            expected_state_action_values = (self.algo_param.gamma*next_state_action_values).unsqueeze(1) + reward.unsqueeze(1) + self.algo_param.alpha*torch.abs(effective_log_ratio.unsqueeze(1))
            if self.T > 10000:
                if self.T%1000==0:
                    print(torch.sum(reward.unsqueeze(1)))
                    print(torch.sum(effective_log_ratio.unsqueeze(1)))
                    print(torch.sum(log_ratio.unsqueeze(1)))
                    print("---------------------------------------------")

        elif self.distance == "Jeffrey":
            expected_state_action_values = (self.algo_param.gamma * next_state_action_values).unsqueeze(1) + reward.unsqueeze(1) + \
                                           self.algo_param.alpha * (
                                               (effective_ratio.unsqueeze(1) - 1)*effective_log_ratio.unsqueeze(1))

        elif self.distance == "Pearson":
            expected_state_action_values = (self.algo_param.gamma * next_state_action_values).unsqueeze(
                1) + reward.unsqueeze(1) + \
                                           self.algo_param.alpha * (
                                                   (effective_ratio.unsqueeze(1) - 1)**2)
        elif self.distance == "Exponential":
            expected_state_action_values = (self.algo_param.gamma * next_state_action_values).unsqueeze(
                1) + reward.unsqueeze(1) + \
                                           self.algo_param.alpha * (
                                                   effective_log_ratio.unsqueeze(1) ** 2)
        if self.distance == "A":
            #expected_state_action_values = (self.algo_param.gamma * next_state_action_values).unsqueeze(
            #    1) + reward.unsqueeze(1) + self.algo_param.alpha * torch.abs(effective_ratio.unsqueeze(1))

            expected_state_action_values = (self.algo_param.gamma * next_state_action_values).unsqueeze(
                1) + reward.unsqueeze(1) #+ self.algo_param.alpha * torch.abs(ratio.unsqueeze(1))

            if self.T > 10000:
                if self.T%1000==0:
                    print(torch.sum(reward.unsqueeze(1)))
                    print(torch.sum(effective_ratio.unsqueeze(1)))
                    print(torch.sum(ratio.unsqueeze(1)))
                    print("---------------------------------------------")

        loss = self.loss_function( state_action_values, expected_state_action_values)

        self.Q_optim[z].zero_grad()
        loss.backward()
        self.Q_optim[z].step()




    def hard_update(self, z):
        self.Target_Q[z].load_state_dict(self.Q[z].state_dict())
    def initalize(self, R_max):

        #inital_phase train after this by continuing with step and train at single iteration and hard update at update interval
        self.steps_done = 0
        self.epsilon = 0.99
        self.T = 0
        state = self.env.reset()
        self.inital_state = state

        """
        for i in range(self.batch_size):
            state = self.step(R_max, self.current_index)
            self.eval(1, R_max, self.current_index)
        
        """
        return state



    def sample_for_Q(self, batch_size, current_z_index):
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


    def save(self, z, q_path, target_q_path, nu_path1, nu_path2):
        self.Q[z].save(q_path)
        self.Target_Q[z].save(target_q_path)


    def load(self, z, q_path, target_q_path, nu_path1, nu_path2):
        self.Q[z].load(q_path)
        self.Target_Q[z].load(target_q_path)

