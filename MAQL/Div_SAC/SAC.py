import numpy as np
import torch


from model import Q_Function_Z_NN, Continuous_Gaussian_Policy
from memory import Replay_Memory


from parameters import Algo_Param, NN_Paramters, Save_Paths, Load_Paths

class SAC():

    def __init__(self, env, q_nn_param, policy_nn_param, algo_nn_param, max_episodes =100, memory_capacity =10000,
                 batch_size=400, save_path = Save_Paths(), load_path= Load_Paths(), action_space = None, alpha_lr=0.00,
                 num_z=2,):
        
        
        self.env = env
        #self.env_type = env_type
        self.device = q_nn_param.device
        self.inital_state = None

        self.num_z = num_z


        self.alpha_lr = alpha_lr
        self.q_nn_param = q_nn_param
        self.policy_nn_param = policy_nn_param
        self.algo_nn_param = algo_nn_param


        self.gamma = self.algo_nn_param.gamma
        self.alpha = self.algo_nn_param.alpha
        self.tau   = self.algo_nn_param.tau

        self.max_episodes = max_episodes
        self.steps_done = 0   #total no of steps done
        self.steps_per_eps = 0 #this is to manually enforce max eps length
        self.update_no = 0
        self.batch_size = batch_size

        self.target_update_interval = self.algo_nn_param.target_update_interval
        self.automatic_alpha_tuning = self.algo_nn_param.automatic_alpha_tuning

        self.critic_1 = {}
        self.critic_2 = {}
        self.policy = {}
        self.critic_target_1 = {}
        self.critic_target_2 = {}
        self.critic_1_optim = {}
        self.critic_2_optim = {}
        self.policy_optim = {}

        self.target_entropy = {}
        self.log_alpha = {}
        self.alpha_optim = {}


        for i in range(self.num_z):

            self.critic_1[i] = Q_Function_Z_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)
            self.critic_2[i] = Q_Function_Z_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)

            self.critic_target_1[i] = Q_Function_Z_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)
            self.critic_target_2[i] = Q_Function_Z_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)

            self.policy[i] = Continuous_Gaussian_Policy(policy_nn_param, save_path=save_path.policy_path,
                                                        load_path=load_path.policy_path, action_space=action_space)

            self.critic_target_1[i].load_state_dict(self.critic_1[i].state_dict())
            self.critic_target_2[i].load_state_dict(self.critic_2[i].state_dict())

            self.critic_1_optim[i] = torch.optim.Adam(self.critic_1[i].parameters(), self.q_nn_param.l_r)
            self.critic_2_optim[i] = torch.optim.Adam(self.critic_2[i].parameters(), self.q_nn_param.l_r)
            self.policy_optim[i] = torch.optim.Adam(self.policy[i].parameters(), self.q_nn_param.l_r)

            if self.automatic_alpha_tuning is True:
                self.target_entropy[i] = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
                self.log_alpha[i] = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim[i] = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        
        self.replay_buffer = Replay_Memory(capacity=memory_capacity)

        self.current_z_index = 0
        self.t = 0
        
    def get_action(self, state, evaluate=False):

        action, log_prob, action_mean = self.policy[self.current_z_index].sample(state, format="torch")

        if evaluate == False:
            return action.cpu().detach().numpy(), action_mean.cpu().detach().numpy()
        else:
            return action_mean.cpu().detach().numpy()

    def initalize(self):

        # inital_phase train after this by continuing with step and train at single iteration and hard update at update interval
        self.steps_done = 0
        self.steps_per_eps = 0
        state = self.env.reset()

        self.inital_state = state
        self.t +=1
        for i in range(self.batch_size):
            state = self.step(state)
            self.t += 1
        return state

    def update(self, batch_size=None):

        if batch_size == None:
            batch_size = self.batch_size
        if batch_size > len(self.replay_buffer):
            return

        self.update_no += 1


        batch = self.replay_buffer.sample(batch_size=batch_size)

        state_batch = batch.state
        action_batch = batch.action
        next_state_batch = batch.next_state
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.q_nn_param.device)
        done_mask_batch = torch.FloatTensor(batch.done_mask).unsqueeze(1).to(self.q_nn_param.device)


        with torch.no_grad():
            next_action_batch, next_log_prob_batch, _ = self.policy[self.current_z_index].sample(next_state_batch, format="torch")
            q1_next_target = self.critic_target_1[self.current_z_index].get_value(next_state_batch, next_action_batch, format="torch")
            q2_next_target = self.critic_target_2[self.current_z_index].get_value(next_state_batch, next_action_batch, format="torch")
            min_q_target = torch.min(q1_next_target, q2_next_target) - self.alpha*next_log_prob_batch
            next_q_value = reward_batch + done_mask_batch*self.gamma*min_q_target

        q1 = self.critic_1[self.current_z_index].get_value(state_batch, action_batch)
        q2 = self.critic_2[self.current_z_index].get_value(state_batch, action_batch)

        q1_loss = 0.5*torch.nn.functional.mse_loss(q1, next_q_value)
        q2_loss = 0.5*torch.nn.functional.mse_loss(q2, next_q_value)


        self.critic_1_optim[self.current_z_index].zero_grad()
        q1_loss.backward()
        self.critic_1_optim[self.current_z_index].step()

        self.critic_2_optim[self.current_z_index].zero_grad()
        q2_loss.backward()
        self.critic_2_optim[self.current_z_index].step()


        pi, log_pi, _ = self.policy[self.current_z_index].sample(state_batch)


        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim[self.current_z_index].zero_grad()
            alpha_loss.backward()
            self.alpha_optim[self.current_z_index].step()

            self.alpha = self.log_alpha[self.current_z_index].exp().detach()


        q1_pi = self.critic_1[self.current_z_index].get_value(state_batch, pi)
        q2_pi = self.critic_2[self.current_z_index].get_value(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)



        policy_loss = ((self.alpha*log_pi) - min_q_pi).mean()

        self.policy_optim[self.current_z_index].zero_grad()
        policy_loss.backward()
        self.policy_optim[self.current_z_index].step()



        if self.update_no%self.target_update_interval == 0:

            self.soft_update(self.critic_target_1[self.current_z_index], self.critic_1[self.current_z_index], self.tau)
            self.soft_update(self.critic_target_2[self.current_z_index], self.critic_2[self.current_z_index], self.tau)


    def step(self, state, random=False):
        batch_size = 1  #since step is for a single sample

        if random:
            action = self.env.action_space.sample()
            action_mean = action
        else:
            action, action_mean = self.get_action(state, evaluate=False)

        next_state, reward, done, _ = self.env.step(action)
        self.t += 1
        self.steps_done += 1
        self.steps_per_eps += 1

        if done:
            mask = 0.0
            self.replay_buffer.push(state, action, action_mean, reward, next_state, self.inital_state, mask, self.t, False)
            next_state = self.env.reset()
            self.steps_per_eps = 0
            return next_state

        if self.steps_per_eps == self.max_episodes:
            mask = 1.0
            self.replay_buffer.push(state, action, action_mean, reward, next_state, self.inital_state, mask, self.t, False)
            next_state = self.env.reset()
            self.steps_per_eps = 0
            return next_state
        mask = 1.0

        self.replay_buffer.push(state, action, action_mean, reward, next_state, self.inital_state, mask, self.t, False)
        return next_state

    def hard_update(self):
        self.critic_target_1[self.current_z_index].load_state_dict(self.critic_1[self.current_z_index].state_dict())
        self.critic_target_2[self.current_z_index].load_state_dict(self.critic_2[self.current_z_index].state_dict())

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save(self, critic_1_path="critic_1", critic_2_path="critic_2",
             critic_1_target_path = "critic_1_target", critic_2_target_path = "critic_2_target",
             policy_path= "policy_target"):

        self.critic_1[self.current_z_index].save(critic_1_path)
        self.critic_2[self.current_z_index].save(critic_2_path)
        self.critic_target_1[self.current_z_index].save(critic_1_target_path)
        self.critic_target_2[self.current_z_index].save(critic_2_target_path)
        self.policy[self.current_z_index].save(policy_path)

    def load(self, critic_1_path="critic_1", critic_2_path="critic_2",
             critic_1_target_path = "critic_1_target", critic_2_target_path = "critic_2_target",
             policy_path= "policy_target"):

        self.critic_1[self.current_z_index].load(critic_1_path)
        self.critic_2[self.current_z_index].load(critic_2_path)
        self.critic_target_1[self.current_z_index].load(critic_1_target_path)
        self.critic_target_2[self.current_z_index].load(critic_2_target_path)
        self.policy[self.current_z_index].load(policy_path)