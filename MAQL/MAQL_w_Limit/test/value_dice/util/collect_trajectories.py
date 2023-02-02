import numpy as np
from util.trajectory_data import Trajectory_data

#https://github.com/google-research/google-research/tree/master/dual_dice

def collect_data(
    env, policy,
    num_trajectories, trajectory_length, gamma = 0.99,
    reward_fn = None):
  """Creates off-policy dataset by running a behavior policy in an environment.

  Args:
    env: An environment.
    policy: A behavior policy.
    num_trajectories: Number of trajectories to collect.
    trajectory_length: Desired length of each trajectory; how many steps to run
      behavior policy in the environment before resetting.
    gamma: Discount used for total and average reward calculation.
    reward_fn: A function (default None) in case the environment reward
      should be overwritten. This function should take in the environment
      reward and the environment's `done' flag and should return a new reward
      to use. A new reward function must be passed in for environments that
      terminate, since the code assumes an infinite-horizon setting.

  Returns:
    data: A TrajectoryData object containing the collected experience.
    avg_episode_rewards: Compute per-episode discounted rewards averaged over
      the trajectories.
    avg_step_rewards: Computed per-step average discounted rewards averaged
      over the trajectories.

  Raises:
    ValueError: If the environment terminates and a reward_fn is not passed in.
  """
  trajectories = []
  trajectory_rewards = []
  total_mass = 0  # For computing average per-step reward.
  for t in range(num_trajectories):
    trajectory = []
    total_reward = 0
    discount = 1.0
    state = env.reset()
    state = np.array([state])
    for _ in range(trajectory_length):

      action = policy.sample(state, format="numpy")[0]

      if env.discrete_action:
        # this block is only for discrete action
        action_scaler = 0
        for i in range(action.shape[0]):
          if action[i] == 1:
            action_scaler = i
        next_state, reward, done, _ = env.step(action_scaler)
      else:
        next_state, reward, done, _ = env.step(action)


      if reward_fn is not None:
        reward = reward_fn(reward, done)
      elif done:
        raise ValueError(
            'Environment terminated but reward_fn is not specified.')

      trajectory.append((state[0], action, reward, next_state))
      total_reward += reward * discount
      total_mass += discount

      next_state = np.array([next_state])
      state = next_state
      discount *= gamma

    trajectories.append(trajectory)
    trajectory_rewards.append(total_reward)
    avg_step_rewards = np.sum(trajectory_rewards) / total_mass

  avg_episode_rewards = np.mean(trajectory_rewards)
  avg_step_rewards = np.sum(trajectory_rewards) / total_mass

  return (Trajectory_data(trajectories, policy=policy),
          avg_episode_rewards, avg_step_rewards)