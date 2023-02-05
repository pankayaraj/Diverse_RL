import torch



def get_policy_ratio(policy1, policy2, state, action, eps=0.0001):

    prob1 = policy1.get_probability(state, action, format="numpy")
    prob2 = policy2.get_probability(state, action, format="numpy")

    return (prob1 + eps)/(prob2 + eps)