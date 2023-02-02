import torch
import numpy as np

from main.maql_sc import MAQL_SC
from main.maql_hc import MAQL_HC
from main.maql_le import MAQL_LE

from q_learning.q_learning import Q_learning

from model import NN_Paramters
from parameters import Algo_Param, Save_Paths, Load_Paths

from env.gridworld.environments import GridWalk
from env.gridworld.enviornment2 import GridWalk2
from env.gridworld.optimal_policy import Optim_Policy_Gridwalk

from util.q_learning_to_policy import Q_learner_Policy
from util.collect_trajectories import collect_data
from util.density_ratio import get_policy_ratio
from main.memory import Transition_tuple, Replay_Memory


#device = torch.device("cuda:0")
device = torch.device("cpu")
policy_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device= device)
nu_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device=device, l_r=0.0001)
q_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=device, l_r=0.05)
algo_param = Algo_Param(hard_update_interval=1)
algo_param.gamma = 0.9


grid_size = 10
env = GridWalk(grid_size, False)
env2 = GridWalk(grid_size, False)
env_tar = GridWalk(grid_size, False)

behaviour_Q  = Q_learning(env, q_param, algo_param)
behaviour_policy = Q_learner_Policy(behaviour_Q.Q, q_param)
behaviour_policy.Q.load("q")

#current_version = "soft_constraint"
#current_version = "hard_constraint"
current_version = "constraint_in_error"

if current_version == "hard_constraint":
    M = MAQL_HC(env, q_param, nu_param, algo_param)
elif current_version == "soft_constraint":
    M = MAQL_SC(env, q_param, nu_param, algo_param)
elif current_version == "constraint_in_error":
    M = MAQL_LE(env, q_param, nu_param, algo_param)

max_episode = 100
no_iteration = 100000
update_interval = 1000
save_interval = 1000
eval_interval = 100

M.initalize()
#initalize the ratio memeory
for k in range(1000):
    s = env_tar.reset()
    t = 0
    i_s = s
    for j in range(max_episode):

        a = behaviour_policy.sample(np.array([s]), "numpy")[0]
        if env_tar.discrete_action:
            # this block is only for discrete action
            a_s = 0
            for i in range(a.shape[0]):
                if a[i] == 1.:
                    a_s = i
            n_s, r, d, _ = env_tar.step(a_s)
        else:
            n_s, r, d, _ = env_tar.step(a)

        if d:
            t = 0
            break
        elif j == max_episode-1:
            t = 0
            break

        M.push_ratio_memory(s, a, r, n_s, i_s, t)
        t += 1
        s = n_s

print("initalization_done")
state = env.reset()
s = env_tar.reset()
i_s = s
t = 0


from main.last_episode_memory import Last_Episode_Container
L = Last_Episode_Container()

for i in range(no_iteration):

    #uploading the behaviour policy stuff
    a  = behaviour_policy.sample(np.array([s]), "numpy")[0]
    if env_tar.discrete_action:
        # this block is only for discrete action
        a_s = 0
        for k in range(a.shape[0]):
            if a[k] == 1:
                a_s =k
        n_s, r, d, _ = env_tar.step(a_s)
    else:
        n_s, r, d, _ = env_tar.step(a)


    M.push_ratio_memory(s, a, r, n_s, i_s, t)
    t += 1
    s = n_s
    if d:
        s = env_tar.reset()
        i_s = s
        t = 0
    elif t == max_episode - 1:
        s = env_tar.reset()
        i_s = s
        t = 0


    M.train(log_ratio_update=True)
    #M.train_log_ratio()
    state = M.step(state)


    if i%update_interval == 0:
        M.hard_update()
        if i != 0:
            #for j in range(20000):
            #    if j%1000== 0:
            print("log_update: learning_rate = " + str(M.log_ratio.current_lr) + " " + str(M.L) )
            #M.train_log_ratio()

        #reset exploration at hard_update
        M.memory = Replay_Memory(capacity=M.memory_capacity)
        M.initalize()


    if i%save_interval == 0:
        print("saving")
        M.save("q_val/q_"+ str(i), "tar_q_val/target_q_"+ str(i), "nu_fun/nu_"+ str(i))
        torch.save(M.memory, "memory/mem")
        torch.save(M.log_ratio_memory, "memory/l_mem")

    if i%eval_interval == 0:
        s_t= env2.reset()
        i_s_t = s_t
        t_ = 0
        rew = 0
        rew_beh = 0
        rew_tot = 0
        log_tot_t = 0

        for _ in range(max_episode):

            a_t = behaviour_policy.sample(np.array([s_t]), "numpy")[0]
            if env2.discrete_action:
                # this block is only for discrete action
                a_s = 0
                for k in range(a_t.shape[0]):
                    if a_t[k] == 1:
                        a_s = k
                n_s_t, r_t, d_t, _ = env2.step(a_s)
            else:
                n_s_t, r_t, d_t, _ = env2.step(a_t)

            #print(behaviour_policy.Q.get_value(np.array([s_t])), a_t, a_s)
            rew_beh += r_t
            s_t = n_s_t
            if d_t == True:
                break
        print("--------------------------------------------------------------------------------------------------------------------------------------")
        print("reward of behaviour at itr " + str(i) + " = " + str(rew_beh) + " at state: " + str(s_t) + " starting from: " + str(
            i_s_t) + " at iter " + str(i))

        s_t = i_s_t
        env2._x = s_t[0]
        env2._y = s_t[1]

        t_ = 0
        for l in range(max_episode):

            t_ += 1
            a_t = M.get_action(s_t)
            n_s_t, r_t, d_t, _ = env2.step(a_t)
            rew += r_t

            a_t_temp = [0 for i in range(q_param.action_dim)]
            a_t_temp[a_t] = 0
            data_ = Transition_tuple([s_t], [a_t_temp], [r_t], [n_s_t], [i_s_t], [t_])
            log = M.get_log_ratio(data_).cpu().detach().numpy()[0][0]
            log_tot_t += log
            rew_tot += r_t - algo_param.alpha*log

            if d_t == True or l == max_episode-1:
                L.push(s_t, a_t_temp, r_t, n_s_t, i_s_t, t_, end_of_eps=True)
            else:
                L.push(s_t, a_t_temp, r_t, n_s_t, i_s_t, t_, end_of_eps=False)


            s_t = n_s_t



            if d_t == True:
                t_ = 0
                break
            elif l == max_episode-1:
                t_ = 0
        print("reward at itr " + str(i) + " = " + str(rew) + " at state: " + str(s_t) + " starting from: " + str(i_s_t) + " at iter " + str(i) + " epsilon = " + str(M.epsilon))
        print("reward_total = " + str(rew_tot) + " log totoal = " + str(log_tot_t) + " log_ratio_est = " + str(M.L) + " neg_KL = " + str(M.log_ratio.current_KL))

        targ_p = M.get_target_policy(target=True)
        #print(M.log_ratio.check_for_limit(L.sample(), targ_p))