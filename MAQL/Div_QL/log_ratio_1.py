from model import Nu_NN_1
import torch
import numpy as np
from math import exp

class Algo_Param():
    def __init__(self, gamma=0.9):
        self.gamma = gamma


class Log_Ratio():

    def __init__(self, nu_param, algo_param, num_z, save_path = "temp", load_path="temp",):

        self.nu_param = nu_param
        self.algo_param = algo_param
        self.num_z = num_z

        #log_ratio estimator
        self.nu_network = Nu_NN_1(nu_param, save_path=save_path, load_path=load_path,  num_z=num_z)
        self.nu_optimizer = torch.optim.Adam(self.nu_network.parameters(), lr=self.nu_param.l_r)

        self.nu_base_lr = self.nu_param.l_r
        self.current_lr = self.nu_base_lr





        self.current_KL = 0
        # exponential fucntion
        self.f = lambda x: torch.exp(x)


    def train_ratio(self, data1, data2, z_arr):

            self.debug_V = {"exp":None, "log_exp":None}

            state1 = data1.state
            action1 = data1.action

            state2 = data2.state
            action2 = data2.action

            weight1 = torch.Tensor(self.algo_param.gamma ** data1.time_step).to(self.nu_param.device)
            weight2 = torch.Tensor(self.algo_param.gamma ** data2.time_step).to(self.nu_param.device)

            # reshaping the weight tensor to facilitate the elmentwise multiplication operation
            no_data1 = weight1.size()[0]
            no_data2 = weight2.size()[0]

            weight1 = torch.reshape(weight1, [no_data1, 1])
            weight2 = torch.reshape(weight2, [no_data2, 1])

            nu1 = self.nu_network(state1, action1)
            nu2 = self.nu_network(state2, action2)

            unweighted_nu_loss_1 = self.f(nu1)
            unweighted_nu_loss_2 = nu2

            loss_1 = torch.log(torch.sum(weight1 * unweighted_nu_loss_1) / torch.sum(weight1))
            loss_2 = torch.sum(weight2 * unweighted_nu_loss_2) / torch.sum(weight2)

            self.debug_V["exp"] = torch.sum(weight1 * unweighted_nu_loss_1) / torch.sum(weight1)
            self.debug_V["log_exp"] = loss_1
            self.debug_V["linear"] = loss_2

            loss = loss_1 - loss_2

            self.nu_optimizer.zero_grad()
            loss.backward()
            self.nu_optimizer.step()

    def debug(self):
        return self.debug_V["exp"], self.debug_V["log_exp"], self.debug_V["linear"]



    def  get_log_state_action_density_ratio(self, data, z_arr):
        state = data.state
        action = data.action



        nu = self.nu_network(state, action)
        return nu
