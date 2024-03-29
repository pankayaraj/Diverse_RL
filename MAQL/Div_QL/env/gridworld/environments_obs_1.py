# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple grid-world environment.

The task here is to walk to the (max_x, max_y) position in a square grid.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np

from typing import Any, Dict, Tuple, Union


class GridWalk(object):
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
    self._n_state = length ** 2
    self._n_action = 4
    self._target_x = int(length/2) - 1
    self._target_y = length - 1
    self.discrete_action = True
    self.current_state = np.array([self._x, self._y])
    self.traget_state = np.array([self._target_x, self._target_y])


    layout = """\
wwwwwwwwwwwww
w f   w   f w
w     w f   w
w  f     f  w
w f   w     w
w     w  f  w
ww wwww f   w
w f   www www
w     w fff w
wf    wfffffw
w      ff   w
w f   w  f  w
wwwww
"""

    obstacle1 = [[3,3], [3,4], [3,5],
                     [4,3], [4,4], [4,5],
                     [5,3], [5,4], [5,5]
                     ]
    
    obstacle2 = [
                  [4,4],

                    ]

    obstacle3 = [[2, 3], [2, 4], [2, 5],
                 [3, 3], [3, 4], [3, 5],
                 [4, 3], [4, 4], [4, 5],
                 [5, 3], [5, 4], [5, 5],
                 [6, 3], [6, 4], [6, 5],
                 [7, 3], [7, 4], [7, 5]
                 ]
    self.obstacle = obstacle3


    self.frozen = np.array([list(map(lambda c: 1 if c == 'f' else 0, line)) for line in layout.splitlines()])


  def reset(self):
    """Resets the agent to a random square."""
    #self._x = np.random.randint(self._length)
    #self._y = np.random.randint(self._length)
    #self._x = 0
    #self._y = 0
    self._x = int(self._length/2) - 1
    self._y = 0

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

    state = [self._x, self._y]
    if state in self.obstacle:
      pass
    else:
      if action == 0:
        if self._x < self._length - 1:
          self._x += 1
        else:
          self._x -= 1
      elif action == 2:
        if self._y < self._length - 1:
          self._y += 1
        else:
          self._y -= 1
      elif action == 1:
        if self._x > 0:
          self._x -= 1
        else:
          self._x += 1
      elif action == 3:
        if self._y > 0:
          self._y -= 1
        else:
          self._y += 1

    #elif action == 4:
    #  pass

      else:
        raise ValueError('Invalid action %s.' % action)
    taxi_distance = (np.abs(self._x - self._target_x) +
                     np.abs(self._y - self._target_y))

    done = False
    #reward = np.exp(-2 * taxi_distance / self._length)
    if self._x == self._target_x and self._y == self._target_y:
      reward = 20
      done = True
    else:
      reward = np.exp(-2 * taxi_distance / self._length)
      #reward = 0
      reward -= 1


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
