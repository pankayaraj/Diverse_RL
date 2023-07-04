import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''
Contains

Policies that samples action and the corresponding action's probabilty and log probabilty
Value functions and action value functions
Nu and Zeta values
'''

#for soft actor critic
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weight_initialize(layer, name):
    if name == 'xavier':
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight, gain=1)
    if name == 'zero':
        torch.nn.init.constant_(layer.weight, 0)


def bias_initialize(layer, name):
    if name == 'zero':
        torch.nn.init.constant_(layer.bias, 0)


class NN_Paramters():
    def __init__(self, state_dim, action_dim, non_linearity = F.tanh, weight_initializer = 'xavier', bias_initializer = 'zero',
                 hidden_layer_dim = [128, 128], device= torch.device('cuda'), l_r=0.0001):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.non_linearity = non_linearity
        self.l_r = l_r


        self.device = device


class BaseNN(nn.Module):

    '''
    Base Neural Network function to inherit from
    save_path       : default path for saving neural network weights
    load_path       : default path for loading neural network weights
    '''

    def __init__(self, save_path, load_path):
        super(BaseNN, self).__init__()

        self.save_path = save_path
        self.load_path = load_path

    def weight_init(self, layer, w_initalizer, b_initalizer):
        #initalize weight
        weight_initialize(layer, w_initalizer)
        bias_initialize(layer, b_initalizer)

    def save(self, path=None):
        #save state dict
        if path is None:
            path = self.save_path
        torch.save(self.state_dict(), path)

    def load(self, path=None):
        #load state dict
        if path is None:
            path = self.load_path
        self.load_state_dict(torch.load(path))


class Continuous_Gaussian_Policy(BaseNN):
    # adapted from https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/model.py

    def __init__(self, nn_params, save_path, load_path, action_space=None):

        super(Continuous_Gaussian_Policy, self).__init__(save_path=save_path, load_path=load_path)

        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity

        self.batch_size = None
        # Hidden layers
        layer_input_dim = self.nn_params.state_dim + 1

        hidden_layer_dim = self.nn_params.hidden_layer_dim
        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer)
            self.layers.append(l)
            layer_input_dim = dim

        # Final Layer
        self.mean = nn.Linear(layer_input_dim, self.nn_params.action_dim)
        self.weight_init(self.mean, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.log_std = nn.Linear(layer_input_dim, self.nn_params.action_dim)
        self.weight_init(self.log_std, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.).to(self.nn_params.device)
            self.action_bias = torch.tensor(0.).to(self.nn_params.device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(self.nn_params.device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(self.nn_params.device)

        self.gaussian = None

    def forward(self, state, z_arr):

        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(self.nn_params.device)
        if type(z_arr) != torch.Tensor:
            z_arr = torch.Tensor(z_arr).to(self.nn_params.device)

        self.batch_size = state.size()[0]
        
        
        inp =  torch.cat((state, z_arr), dim= 1)

        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)

        mean = self.mean(inp)
        log_std = self.log_std(inp)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample_with_std(self, state, z_arr, format="torch"):
        mean, log_std = self.forward(state=state, z_arr=z_arr)
        std = log_std.exp()

        gaussian = torch.distributions.Normal(loc=mean, scale=std)

        # sample for reparametrization trick
        x_t = gaussian.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = gaussian.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)

        if len(log_prob.shape) != 1:
            log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        if format == "torch":
            return action, log_prob, mean, std
        else:
            return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy(), mean.cpu().detach().numpy(), std.cpu().detach().numpy()

    def sample(self, state, format="torch"):

        mean , log_std = self.forward(state=state)
        std = log_std.exp()

        gaussian = torch.distributions.Normal(loc=mean, scale=std)
        self.gaussian = gaussian #to use for IRM sampling later on

        #sample for reparametrization trick
        x_t = gaussian.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = gaussian.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)


        if len(log_prob.shape) != 1:
            log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        self.x_t = x_t

        if format == "torch":
            return action, log_prob, mean
        else:
            return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy(), mean.cpu().detach().numpy()
   

    def to(self, device):
        super().to(device)
        self.nn_params.device= device

class Q_Function_Z_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path):

        super(Q_Function_Z_NN, self).__init__(save_path=save_path, load_path=load_path)
        
        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity

        # Hidden layers
        layer_input_dim = self.nn_params.state_dim + self.nn_params.action_dim + 1
        hidden_layer_dim = self.nn_params.hidden_layer_dim
        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer)
            self.layers.append(l)
            layer_input_dim = dim

        #Final Layer
        self.Q_value = nn.Linear(layer_input_dim, 1)
        self.weight_init(self.Q_value, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

    def forward(self, state, action):
        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(self.nn_params.device)
        if type(action) != torch.Tensor:
            action = torch.Tensor(action).to(self.nn_params.device)

        inp = torch.cat((state, action), dim= 1)

        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)
        Q_s_a = self.Q_value(inp)

        return Q_s_a

    def get_value(self, state, action, format="torch"):

        if format == "torch":
            return self.forward(state, action)
        elif format == "numpy":
            return  self.forward(state, action).cpu().detach().numpy()

    def to(self, device):
        super().to(device)
        self.nn_params.device= device



class Nu_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path, num_z, state_action=True):
        super(Nu_NN, self).__init__(save_path=save_path, load_path=load_path)
        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity
        self.state_action = state_action
        # Hidden layers
        if state_action:
            layer_input_dim = self.nn_params.state_dim + self.nn_params.action_dim + num_z
        else:
            layer_input_dim = self.nn_params.state_dim + num_z
        hidden_layer_dim = self.nn_params.hidden_layer_dim
        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer)
            self.layers.append(l)
            layer_input_dim = dim

        # Final Layer
        self.nu = nn.Linear(layer_input_dim, 1)
        self.weight_init(self.nu, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

    def forward(self, state, z, action):

        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(self.nn_params.device)
        if type(z) != torch.Tensor:
            z = torch.Tensor(z).to(self.nn_params.device)
        if type(action) != torch.Tensor:
            action = torch.Tensor(action).to(self.nn_params.device)

        
        if z.dim() == 1:
            state = torch.cat((state, z), dim=0)
        else:
            state = torch.cat((state, z), dim=1)


        if self.state_action:
            if type(action) != torch.Tensor:
                action = torch.Tensor(action).to(self.nn_params.device)

            if len(action.shape.item()) == 1:
                print(state, action)
                inp = torch.cat((state, action), dim=0)
            else:
                inp = torch.cat((state, action), dim=1)
        else:
            inp = state
        
        
        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)
        NU = self.nu(inp)

        return NU
        #NU = torch.clamp(self.nu(inp), 70, -70)




class Zeta_NN(BaseNN):
    """
        state_action : weather to estimate for state action or just state.
    """

    def __init__(self, nn_params, save_path, load_path, num_z, state_action=True):
        super(Zeta_NN, self).__init__(save_path=save_path, load_path=load_path)
        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity
        self.state_action = state_action

        if state_action:
            layer_input_dim = self.nn_params.state_dim + self.nn_params.action_dim +  num_z
        else:
            layer_input_dim = self.nn_params.state_dim + + num_z

        hidden_layer_dim = self.nn_params.hidden_layer_dim

        # Hidden layers
        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer)
            self.layers.append(l)
            layer_input_dim = dim

        # Final Layer
        self.zeta = nn.Linear(layer_input_dim, 1)
        self.weight_init(self.zeta, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

    def forward(self, state, z, action):
        """ Here the input can either be the state or a concatanation of state and action"""
        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(self.nn_params.device)
        if type(z) != torch.Tensor:
            z = torch.Tensor(z).to(self.nn_params.device)

        if z.dim() == 1:
            state = torch.cat((state, z), dim= 0)
        else:
            state = torch.cat((state, z), dim=1)

        if self.state_action:
            if type(action) != torch.Tensor:
                action = torch.Tensor(action).to(self.nn_params.device)
            inp = torch.cat((state, action), dim=1)
        else:
            inp =state

        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)
        Zeta = self.zeta(inp)

        return Zeta



