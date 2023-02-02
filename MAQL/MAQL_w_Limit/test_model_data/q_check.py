import numpy as np
import torch
import gym
from env.gridworld.environments import GridWalk
from main.creator_functions import get_log_ratio, get_maql, get_q_learning, get_q_learning_policy

gamma = 0.995

#env = gym.make("CartPole-v0")
grid_size = 10
env = GridWalk(grid_size, False)
env_b = GridWalk(grid_size, False)

save_interval = 3000

l = get_log_ratio(env)
l.nu_network.load("nu_fun/nu_" + str(save_interval))

Q = get_q_learning(env)
Q.load('q_val/q_' + str(save_interval) ,'tar_q_val/target_q_' + str(save_interval))

Q_behaviour = get_q_learning(env)
Q_behaviour.load("q", "target_q")

target_policy = get_q_learning_policy(Q)

state = env.reset()
env_b.update_curerent_state(state)
state_b = state

print(state, state_b)
rew = 0
rew_b = 0
log = 0
for i in range(100):
    prev_state = state
    a = target_policy.sample(np.array([state]))
    action = Q.get_action(state)

    action_b = Q_behaviour.get_action(state_b)


    state, reward, done, _ = env.step(action)
    state_b, reward_b, done_b, _ = env_b.step(action_b)

    n, nxt_n = l.compute_for_eval(np.array([prev_state]), np.array(a), np.array([state]), target_policy)
    print(n, nxt_n)
    log_i = n - gamma*nxt_n
    print(log_i[0][0], state, action, state_b, action_b)
    log += log_i[0][0]
    rew += reward
    rew_b += reward_b
    if done == True:
        break
    #env.render()
print(state, state_b)
print(rew, rew_b, log)