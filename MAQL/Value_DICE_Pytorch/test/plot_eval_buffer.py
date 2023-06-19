import torch
import numpy as np
from value_dice_eval_buffer import Value_Dice_Eval_Buffer
from dual_dice_eval_buffer import Dice_Eval_Buffer
from value_dice import Algo_Param
from model import NN_Paramters
import matplotlib.pyplot as plt


from distances import KL, Exponential, Pearson, Hellinger, Jeffery, Reyni, Chernoff, Alpha_Beta


algo_param = Algo_Param()
nu_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh,
                        device=torch.device("cpu"), l_r=0.0001)


nu_path = ["nu_1", "nu_2", "nu_4"]

algo_param.gamma = 0.995


D = [Dice_Eval_Buffer( nu_param, algo_param) for _ in range(len(nu_path))]


V = [Value_Dice_Eval_Buffer( nu_param, algo_param) for _ in range(len(nu_path))]

for i in range(len(nu_path)):
    D[i].nu_network.load("nu_eval_buff/dice/" + str(nu_path[i]))
    V[i].nu_network.load("nu_eval_buff/value_dice/" + str(nu_path[i]))

scenarios = [{"KL":0, "Exponential":0, "Pearson":0,
                     "Hellinger":0, "Jeffery":0, "Reyni":0, "Chernoff":0, "Alpha_Beta":0} for _ in range(len(nu_path))]



Buffer = torch.load("behavior_sub_1")
data = Buffer.sample(int(Buffer.no_data/20))
no_data = int(Buffer.no_data/20)

gamma = 0.995
theta = 2
phi = 2

for i in range(len(nu_path)):

    v = V[i]
    d = D[i]

    scenarios[i]["KL"] = KL(v, data, no_data, gamma)
    scenarios[i]["Exponential"] = Exponential(v, data, no_data, gamma)
    scenarios[i]["Pearson"] = Pearson(d, data, no_data, gamma)
    scenarios[i]["Hellinger"] = Hellinger(d, data, no_data, gamma)
    scenarios[i]["Jeffery"] = Jeffery(v, d, data, no_data, gamma)
    scenarios[i]["Reyni"] = Reyni(d, data, no_data, gamma, theta)
    scenarios[i]["Chernoff"] = Chernoff(d, data, no_data, gamma, theta)
    scenarios[i]["Alpha_Beta"]= Alpha_Beta(d, data, no_data, gamma, theta, phi)

for i in range(len(nu_path)):
    print(scenarios[i])


labels = ["KL", "Exponential", "Pearson", "Hellinger", "Jeffery", "Reyni", "Chernoff", "Alpha_Beta"]
#labels = ["Scenario 1", "Scenario 2", "Scenario 3"]
x = np.arange(len(labels))
print(x)
width = 0.1
fig, ax = plt.subplots(figsize=(14.0, 10.0))
ax.tick_params(axis='both', which='major', labelsize=14)

for i in range(len(labels)):


    rects1 = ax.bar(x[i] - 0.3, scenarios[0][labels[i]], 0.3, capsize=7, label="Scenario 1", color="b")
    rects2 = ax.bar(x[i] - 0.0, scenarios[1][labels[i]], 0.3, capsize=7, label="Scenario 2", color="g")
    rects3 = ax.bar(x[i] + 0.3, scenarios[2][labels[i]], 0.3, capsize=7, label="Scenario 3", color="r")

    """
    rects1 = ax.bar(x - 0.8, scenarios[i]["KL"], 0.2, capsize=7,)
    rects2 = ax.bar(x - 0.3, scenarios[i]["Exponential"], 0.2, capsize=7,)
    rects3 = ax.bar(x + 0.0, scenarios[i]["Pearson"], 0.2, capsize=7,)
    rects4 = ax.bar(x + 0.3, scenarios[i]["Hellinger"], 0.2, capsize=7, )
    rects5 = ax.bar(x + 0.6, scenarios[i]["Jeffery"], 0.2, capsize=7, )
    rects6 = ax.bar(x + 0.9, scenarios[i]["Reyni"], 0.2, capsize=7, )
    rects7 = ax.bar(x + 0.9, scenarios[i]["Chernoff"], 0.2, capsize=7, )
    rects8 = ax.bar(x + 0.9, scenarios[i]["Alpha_Beta"], 0.2, capsize=7, )
    """

    handles = ["Scenario 1", "Scenario 2", "Scenario 3", ]
    ax.legend([rects1, rects2, rects3,], handles, prop={'size': 28}, loc='upper center', bbox_to_anchor=(0.5, 1.15),
               fancybox=True, shadow=True, ncol=6)



    ax.set_ylabel('Distance and Divergence', size=45 )
    ax.set_xlabel('Length', size=45)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

for i in x:
    plt.axvline(x=i+0.5)

plt.show()
#plt.savefig("Distances")