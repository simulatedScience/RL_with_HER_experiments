"""
This implements the bitflip problem using the Q-leaerning interface.
"""
from typing import Tuple
from copy import copy

import numpy as np
import tensorflow as tf

from q_problem_interface import Q_learning_problem

class Bitflip_problem(Q_learning_problem):
  def __init__(self, problem_size: int):
    self.problem_size: int = problem_size
    # create goal as tensorflow tensor of ones
    self.goal: tf.Tensor = tf.ones((problem_size,), dtype=tf.int32)

  def gen_start_state(self) -> tf.Tensor:
    """
    generate a random 0-1-array as a start state for the bitflip problem.

    Returns:
        tf.Tensor: the start state
    """
    return tf.random.uniform(shape=(self.problem_size,), minval=0, maxval=1, dtype=tf.int32)
    # return np.random.randint(0, 2, self.problem_size, dtype=np.int8)

  def gen_goal_state(self) -> tf.Tensor:
    """
    generate a random 0-1-array as a goal state for the bitflip problem.
    This goal is saved in the `goal` attribute.

    Returns:
        tf.Tensor: the goal state
    """
    
    self.goal = tf.random.uniform(shape=(self.problem_size,), minval=0, maxval=1, dtype=tf.int32)
    # self.goal = np.random.randint(0, 2, self.problem_size, dtype=np.int8)
    return self.goal

  def get_num_actions(self) -> int:
    """
    get the number of possible actions in the environment.

    Returns:
        int: the number of possible actions
    """
    return self.problem_size
  
  def get_state_size(self) -> Tuple[int]:
    """
    get the size and shape of each state in the state space.

    Returns:
        int: the size of the state space
    """
    return (self.problem_size,)

  def get_reward(self,
        state: tf.Tensor,
        action: int,
        goal: tf.Tensor = None) -> Tuple[float, bool]:
    """
    get the reward for taking an action in a given state.

    Args:
        state (tf.Tensor): the current state
        action (int): the action to take
        goal (tf.Tensor, optional): the goal state. If goal is None, the goal state stored in the `goal` attribute is used. Defaults to None.

    Returns:
        float: the reward for taking the action
        bool: True if the goal state is reached, False otherwise
    """
    if goal is None:
      goal = self.goal
    reward = 1 if tf.reduce_all(tf.equal(state[action], goal[action])) else 0
    goal_reached = tf.reduce_all(tf.equal(state, goal))
    return reward, goal_reached

  def get_next_state(self,
        state: tf.Tensor,
        action: int) -> tf.Tensor:
    """
    get the next state after taking an action in a given state.

    Args:
        state (tf.Tensor): the current state
        action (int): the action to take

    Returns:
        tf.Tensor: the next state
    """
    # tensorflow tensor with 1 at the action index and 0 everywhere else
    action_tensor = tf.one_hot(action, self.problem_size, dtype=tf.int32)
    # flip the bit at the action index
    new_state = tf.abs(state - action_tensor)

    # new_state = copy(state)
    # flip the bit at position `action`
    # new_state[action] = 1 - new_state[action]
    # print(state.numpy(), action, new_state.numpy())
    return new_state

  def is_goal(self, state: tf.Tensor) -> bool:
    """
    check if the given state is the goal state stored in the `goal` attribute.

    Args:
        state (tf.Tensor): the state to check

    Returns:
        bool: True if the state is the goal state, False otherwise
    """
    return tf.reduce_all(tf.equal(state, self.goal))
    # return np.all(state == self.goal)

  def get_actions(self, state: tf.Tensor) -> tf.Tensor:
    """
    get the possible actions for a given state. These are always all integers in range [0, problem_size-1].

    Args:
        state (tf.Tensor): the state to get the actions for

    Returns:
        tf.Tensor: the actions
    """
    return tf.range(self.problem_size)
    # return np.arange(self.problem_size)

  def take_action(self, state: tf.Tensor, action: int) -> Tuple[float, tf.Tensor, bool]:
    """
    take an action in a given state and return the reward and the next state.

    Args:
        state (tf.Tensor): the current state
        action (int): the action to take

    Returns:
        float: the reward for taking the action
        tf.Tensor: the next state after taking the action
        bool: True if the goal state is reached, False otherwise
    """
    reward, goal_reached = self.get_reward(state, action, self.goal)
    next_state = self.get_next_state(state, action)
    return reward, next_state, goal_reached

if __name__ == "__main__":
  # test the bitflip problem
  problem = Bitflip_problem(6)
  print(problem.gen_start_state())
  print(problem.gen_goal_state())
  # print(problem.get_reward(np.array([0, 1, 0, 1]), 1))
  # print(problem.get_next_state(np.array([0, 1, 0, 1]), 1))
  # print(problem.is_goal(np.array([0, 1, 0, 1])))
  # print(problem.get_actions(np.array([0, 1, 0, 1])))
  # print(problem.take_action(np.array([0, 1, 0, 1]), 1))
