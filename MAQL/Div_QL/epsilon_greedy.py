import torch
import numpy as np
import math

def epsilon_greedy(q_values, steps_done, epsilon, action_dim, eps_end= 0.0005, eps_decay=0.9999):

    rand = np.random.random()

    epsilon = max(epsilon * eps_decay, eps_end)
    #epsilon = eps_end + (eps_start - eps_end) * math.exp(-1 * steps_done / eps_decay)
    if rand < epsilon:
        action_scaler = np.random.choice(action_dim, 1)[0]
    else:
        action_scaler = np.argmax(q_values)
    steps_done += 1
    return action_scaler, steps_done, epsilon



