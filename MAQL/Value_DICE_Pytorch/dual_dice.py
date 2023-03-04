from model import Nu_NN
import torch
import numpy as np


class Algo_Param():
    def __init__(self, gamma=0.9):
        self.gamma = gamma


class Dice():

    def __init__(self, target_policy, nu_param, algo_param, deterministic_env=True, averege_next_nu = True,
                 discrete_policy=True, save_path = "temp", load_path="temp" ):

        self.nu_param = nu_param
        self.algo_param = algo_param


        self.nu_network = Nu_NN(nu_param, save_path=save_path, load_path=load_path)
        self.nu_optimizer = torch.optim.Adam(self.nu_network.parameters(), lr=self.nu_param.l_r)

        self.target_policy = target_policy

        self.deterministic_env = deterministic_env
        self.average_next_nu = averege_next_nu
        self.discrete_poliy = discrete_policy

        # exponential fucntion
        self.f = lambda x: torch.abs(x)**2/2

    def train_dice(self, data):

        self.debug_V = {"x**2":None, "linear": None}
        state = data.state
        action = data.action
        next_state = data.next_state
        initial_state = data.initial_state

        weight = torch.Tensor(self.algo_param.gamma**data.time_step).to(self.nu_param.device)

        #reshaping the weight tensor to facilitate the elmentwise multiplication operation
        no_data = weight.size()[0]
        weight = torch.reshape(weight, [no_data, 1])


        next_action = self.target_policy.sample(next_state, format="torch")
        initial_action = self.target_policy.sample(initial_state, format="torch")

        nu, next_nu, initial_nu = self.compute(state, action, next_state, next_action, initial_state, initial_action)


        delt_nu = nu - self.algo_param.gamma*next_nu

        unweighted_nu_loss_1 = self.f(delt_nu)
        unweighted_nu_loss_2 = initial_nu


        loss_1 = torch.sum(weight*unweighted_nu_loss_1)/torch.sum(weight)
        loss_2 = torch.sum(weight*unweighted_nu_loss_2)/torch.sum(weight)

        self.debug_V["x**2"] = torch.sum(weight*unweighted_nu_loss_1)/torch.sum(weight)
        self.debug_V["linear"] = loss_2
        #loss_1 = torch.log(torch.sum(unweighted_nu_loss_1))
        #loss_2 = torch.sum(unweighted_nu_loss_2)

        loss = loss_1 - (1-self.algo_param.gamma)*loss_2

        self.nu_optimizer.zero_grad()
        loss.backward()
        self.nu_optimizer.step()

    def debug(self):
        return self.debug_V["x**2"], self.debug_V["linear"]

    def compute(self, state, action, next_state, next_action, initial_state, initial_action):

        nu = self.nu_network(state, action)
        initial_nu = self.nu_network(initial_state, initial_action)

        #in case of the deterministic env then we can take the averge over all actions n it would suffice. Even in the absese
        #if averge netx nu flag is set we average over all the actions to reduce any bias

        if self.average_next_nu and self.discrete_poliy:
            # This is only for discrete policy. Since the hot vector dim corresponds to the no of actions in this case
            batch_size = torch.Tensor(state).size()[0]

            one_hot_next_action = [torch.zeros ([batch_size, self.nu_param.action_dim])
                                   for _ in range(self.nu_param.action_dim)]
            for action_i in range(self.nu_param.action_dim):
                one_hot_next_action[action_i][:, action_i] = 1

            next_target_probabilities = self.target_policy.get_probabilities(next_state)
            all_next_nu = [self.nu_network(next_state, one_hot_next_action[action_i])
                           for action_i in range(self.nu_param.action_dim)]

            next_nu = torch.zeros([batch_size, 1]).to(self.nu_param.device)
            for action_i in range(self.nu_param.action_dim):
                next_nu += torch.reshape(next_target_probabilities[:, action_i], shape=(batch_size, 1)) * all_next_nu[
                    action_i]

        else:
            next_nu = self.nu_network(next_state, next_action)

        return nu, next_nu, initial_nu

    def get_state_action_density_ratio(self, data):

        initial_action = self.target_policy.sample(data.initial_state)
        next_action = self.target_policy.sample(data.next_state)

        nu, next_nu, inital_nu = self.compute(data.state, data.action, data.next_state, next_action,
                                                  data.initial_state, initial_action)

        return nu - self.algo_param.gamma * next_nu



