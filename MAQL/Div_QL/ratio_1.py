from model import Nu_NN_1
import torch
import numpy as np


class Algo_Param():
    def __init__(self, gamma=0.9):
        self.gamma = gamma


class Ratio():

    def __init__(self, nu_param, algo_param, num_z, state_action=True, save_path = "temp", load_path="temp",):

        self.nu_param = nu_param
        self.algo_param = algo_param
        self.num_z = num_z

        # log_ratio estimator
        self.nu_network = Nu_NN_1(nu_param, save_path=save_path, load_path=load_path, num_z=num_z, state_action=state_action,)
        self.nu_optimizer = torch.optim.Adam(self.nu_network.parameters(), lr=self.nu_param.l_r)

        self.nu_base_lr = self.nu_param.l_r
        self.current_lr = self.nu_base_lr

        #
        self.current_ratio = 0
        self.f = lambda x: torch.abs(x)**2/2

        self.debug_V = {"x**2": None, "linear": None}

    def train_dice(self, data1, data2, z_arr):


        state1 = data1.state
        action1 = data1.action

        state2 = data2.state
        action2 = data2.action



        weight1 = torch.Tensor(self.algo_param.gamma ** data1.time_step).to(self.nu_param.device)
        #weight2 = torch.Tensor(self.algo_param.gamma ** data2.time_step).to(self.nu_param.device)

        # reshaping the weight tensor to facilitate the elmentwise multiplication operation
        no_data1 = weight1.size()[0]
        no_data2 = weight1.size()[0]


        #weight1 = torch.reshape(weight1, [no_data1, 1])
        weight2 = torch.reshape(weight1, [no_data2, 1])

        nu1 = self.nu_network(state1, action1)
        nu2 = self.nu_network(state2, action2)

        offset = 0.01*torch.ones([no_data2, 1])
        #print(offset)


        unweighted_nu_loss_1 = self.f(nu1+offset)
        unweighted_nu_loss_2 = nu2

        #loss_1 = weight1 *torch.sum(unweighted_nu_loss_1)/torch.sum(weight1)
        loss_1 = torch.sum(weight1 * unweighted_nu_loss_1)/ torch.sum(weight1)
        loss_2 = torch.sum(weight1 * unweighted_nu_loss_2) / torch.sum(weight1)

        #loss_1 = torch.sum(unweighted_nu_loss_1)
        #loss_2 = torch.sum(unweighted_nu_loss_2)

        self.debug_V["x**2"] = loss_1
        self.debug_V["linear"] = loss_2

        loss = torch.abs(loss_1 - loss_2)

        self.nu_optimizer.zero_grad()
        loss.backward()
        self.nu_optimizer.step()

    def debug(self):
        return self.debug_V["x**2"], self.debug_V["linear"]
    def get_state_action_density_ratio(self, data, z_arr):
        state = data.state
        action = data.action

        nu = self.nu_network(state, action)
        return nu


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
