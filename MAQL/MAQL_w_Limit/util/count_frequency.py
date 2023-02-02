import torch
import numpy as np




def collect_freqency(data, no_states, grid_size, gamma):

    freq = np.zeros(shape=[no_states,])
    for i in data.iterate_through():

        state = i[0]
        x =  state[0]
        y = state[1]

        xy = x*grid_size + y
        freq[xy] += gamma*i[4]
    return freq

