"""Simple grid-world environment.

The task here is to walk to one of many goal position in a square grid.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np

from typing import Any, Dict, Tuple, Union


class GridWalk2(object):
  """Walk on grid to target location."""

  def __init__(self, length, tabular_obs = True):
    """Initializes the environment.

    Args:
      length: The length of the square gridworld.
      tabular_obs: Whether to use tabular observations. Otherwise observations
        are x, y coordinates.
    """
    self._length = length
    self._tabular_obs = tabular_obs

    self._x = np.random.randint(length)
    self._y = np.random.randint(length)
    #self._x = 0
    #self._y = 7

    self._n_state = length ** 2
    self._n_action = 5

    self._target_x = length - 1
    self._target_y = length - 1

    self._target_x2 = 5
    self._target_y2 = 1


    self.discrete_action = True
    self.current_state = np.array([self._x, self._y])
    self.traget_state = np.array([self._target_x, self._target_y])

  def reset(self):
    """Resets the agent to a random square."""
    self._x = np.random.randint(self._length)
    self._y = np.random.randint(self._length)
    #self._x = 0
    #self._y = 9

    self.current_state = np.array([self._x, self._y])
    return self._get_obs()

  def update_curerent_state(self, current_state):
    self._x = current_state[0]
    self._y = current_state[1]

  def _get_obs(self):
    """Gets current observation."""
    if self._tabular_obs:
      return self._x * self._length + self._y
    else:
      return np.array([self._x, self._y])

  def get_tabular_obs(self, xy_obs):
    """Gets tabular observation given non-tabular (x,y) observation."""
    return self._length * xy_obs[Ellipsis, 0] + xy_obs[Ellipsis, 1]

  def get_xy_obs(self, state):
    """Gets (x,y) coordinates given tabular observation."""
    x = state // self._length
    y = state % self._length
    return np.stack([x, y], axis=-1)

  def step(self, action):
    """Perform a step in the environment.

    Args:
      action: A valid action (one of 0, 1, 2, 3).
      0 : right, 1: left, 2:up, 3:down

    Returns:
      next_obs: Observation after action is applied.
      reward: Environment step reward.
      done: Whether the episode has terminated.
      info: A dictionary of additional environment information.

    Raises:
      ValueError: If the input action is invalid.
    """


    if action == 0:
      if self._x < self._length - 1:
        self._x += 1
    elif action == 2:
      if self._y < self._length - 1:
        self._y += 1

    elif action == 1:
      if self._x > 0:
        self._x -= 1
    elif action == 3:
      if self._y > 0:
        self._y -= 1

    elif action == 4:
      pass

    else:
      raise ValueError('Invalid action %s.' % action)
    taxi_distance = (np.abs(self._x - self._target_x) +
                     np.abs(self._y - self._target_y))

    taxi_distance_2 = (np.abs(self._x - self._target_x2) +
                     np.abs(self._y - self._target_y2))


    reward_1 = np.exp(-2 * taxi_distance/ self._length)
    reward_2 = np.exp(-2 * taxi_distance_2 / self._length)

    reward = reward_1 + 0.2*reward_2
    done = False

    self.current_state = np.array([self._x, self._y])
    return self._get_obs(), reward, done, {}

  def render(self):

    for i in range(self._length):
      for j in range(self._length):
        temp = i*self._length + j
        if temp == self.get_tabular_obs(self.current_state):
          print(' c ', end = "")
        elif temp == self.get_tabular_obs(self.traget_state):
          print(" G ", end= "")
        else:
          print(" _ ", end = "")
      print("\n")

  def render_trajectory(self, trajectory):

    states = []
    for state, _, _, _ in trajectory:
      states.append(self.get_tabular_obs(state))


    for i in range(self._length):
      for j in range(self._length):
        temp = i*self._length + j
        if temp in states:
          print(' p ', end = "")
        elif temp == self.get_tabular_obs(self.traget_state):
          print(" G ", end= "")
        else:
          print(" _ ", end = "")
      print("\n")



  @property
  def num_states(self):
    return self._n_state

  @property
  def num_actions(self):
    return self._n_action

  @property
  def state_dim(self):
    return 1 if self._tabular_obs else 2

  @property
  def action_dim(self):
    return self._n_action
