import torch


#Divergences
def KL(V, data, no_data, gamma=0.995):
    w = torch.reshape(torch.Tensor(gamma ** data.time_step), shape=(no_data, 1))
    nu = -V.get_log_state_action_density_ratio(data)
    return (torch.sum(nu * w) / torch.sum(w)).item()


def Exponential(V, data, no_data, gamma=0.995):
    w = torch.reshape(torch.Tensor(gamma ** data.time_step), shape=(no_data, 1))
    nu = -V.get_log_state_action_density_ratio(data)
    x = nu**2
    return (torch.sum(x * w) / torch.sum(w)).item()

def Pearson(D, data, no_data, gamma=0.995):
    w = torch.reshape(torch.Tensor(gamma ** data.time_step), shape=(no_data, 1))
    nu = D.get_state_action_density_ratio(data)
    x = torch.square(nu - 1)
    return (torch.sum(x * w) / torch.sum(w)).item()

def Hellinger(D, data, no_data, gamma = 0.995):
    w = torch.reshape(torch.Tensor(gamma ** data.time_step), shape=(no_data, 1))
    nu = D.get_state_action_density_ratio(data)

    x = torch.square(torch.sqrt(nu) - 1)
    return (torch.sum(x * w) / torch.sum(w)).item()

def Jeffery(V, D, data, no_data, gamma= 0.995):
    w = torch.reshape(torch.Tensor(gamma ** data.time_step), shape=(no_data, 1))
    nu1 = -V.get_log_state_action_density_ratio(data)
    nu2 = D.get_state_action_density_ratio(data)

    x = (nu2 - 1)*nu1
    return (torch.sum(x * w) / torch.sum(w)).item()

def Reyni(D, data, no_data, gamma=0.995, theta=2):

    w = torch.reshape(torch.Tensor(gamma ** data.time_step), shape=(no_data, 1))
    nu = D.get_state_action_density_ratio(data)
    x = nu**theta-nu
    w_theta = 1/(theta*(theta-1))
    return (w_theta*torch.sum(x * w) / torch.sum(w)).item()

def Chernoff(D, data, no_data, gamma=0.995, theta=2):
    w = torch.reshape(torch.Tensor(gamma ** data.time_step), shape=(no_data, 1))
    nu = D.get_state_action_density_ratio(data)
    x = 1 - nu**((theta+1)/2)
    w_theta = 4 / (1-theta**2)
    return (w_theta * torch.sum(x * w) / torch.sum(w)).item()

def Alpha_Beta(D, data, no_data, gamma=0.995, theta=2, phi=2):
    w = torch.reshape(torch.Tensor(gamma ** data.time_step), shape=(no_data, 1))
    nu = D.get_state_action_density_ratio(data)
    x = (1 - nu ** ((1-theta) / 2))*(1 - nu ** ((1-phi) / 2))
    w_theta_phi = 2 / (1 - theta)*(1 - phi)
    return (w_theta_phi * torch.sum(x * w) / torch.sum(w)).item()

