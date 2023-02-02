from q_learning import DQN_double
from env.gridworld.environments import GridWalk
import torch

grid_size = 10
env = GridWalk(10, False)

n_state = 2
# Number of actions
#n_action = env.action_space.n
n_action = 5
# Number of episodes
episodes = 150
# Number of hidden nodes in the DQN
n_hidden = 50
# Learning rate
lr = 0.001

D = DQN_double(n_state, n_action, n_hidden, lr)
D.model.load_state_dict(torch.load("q"))

state = env.reset()

print(state)
rew = 0
for i in range(200):

    q_values = D.predict(state)
    action = torch.argmax(q_values).item()
    state, reward, done, _ = env.step(action)
    rew += reward
    if done == True:
        break
    #env.render()
print(state)
print(rew)