"""
This module implements an interface for Q-learning problems.
"""
from typing import Tuple

import numpy as np


class Q_learning_problem:
  def __init__(self):
    self.goal: np.ndarray = None
    raise NotImplementedError

  def get_num_actions(self) -> int:
    """
    get the number of possible actions in the environment.

    Returns:
        int: the number of possible actions
    """
    raise NotImplementedError

  def get_state_size(self) -> Tuple[int]:
    """
    get the size and shape of each state in the state space.

    Returns:
        int: the size of the state space
    """
    raise NotImplementedError

  def gen_start_state(self) -> np.ndarray:
    """
    generate a random start state for the problem.

    Returns:
        np.ndarray: the start state
    """
    raise NotImplementedError
  
  def gen_goal_state(self) -> np.ndarray:
    """
    generate a random goal state for the problem and save it as self.goal.

    Returns:
        np.ndarray: the goal state
    """
    raise NotImplementedError
  
  def get_goal(self) -> np.ndarray:
    """
    get the goal state of the problem.

    Returns:
        np.ndarray: the goal state
    """
    return self.goal

  def get_reward(self,
        state: np.ndarray,
        action: int) -> float:
    """
    get the reward for taking an action in a given state.

    Args:
        state (np.ndarray): the current state
        action (int): the action to take

    Returns:
        float: the reward for taking the action
    """
    raise NotImplementedError

  def get_next_state(self,
        state: np.ndarray,
        action: int) -> np.ndarray:
    """
    get the next state after taking an action in a given state.

    Args:
        state (np.ndarray): the current state
        action (int): the action to take

    Returns:
        np.ndarray: the next state
    """
    raise NotImplementedError

  def is_goal(self,
        state: np.ndarray) -> bool:
    """
    check if a given state is a goal state.

    Args:
        state (np.ndarray): the state to check

    Returns:
        bool: True if the state is a goal state, False otherwise
    """
    raise NotImplementedError

  def get_actions(self,
        state: np.ndarray) -> np.ndarray:
    """
    get the possible actions in a given state.

    Args:
        state (np.ndarray): the state to check

    Returns:
        np.ndarray: the possible actions in the state
    """
    raise NotImplementedError

  def take_action(self,
        state: np.ndarray,
        action: int) -> Tuple[float, np.ndarray]:
    """
    take an action in the environment and observe the reward and the next state.

    Args:
        state (np.ndarray): the current state of the environment
        action (int): the action to take

    Returns:
        tuple: the reward for the taken action and the next state
    """
    raise NotImplementedError

  def get_possible_next_states(self, 
        state: np.ndarray) -> np.ndarray:
    """
    get the possible next states after taking any action in a given state.

    Args:
        state (np.ndarray): the state to check

    Returns:
        np.ndarray: the possible next states
    """
    raise NotImplementedError

  def get_possible_next_states_and_rewards(self,
        state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    get the possible next states and the rewards for taking any action in a given state.

    Args:
        state (np.ndarray): the state to check

    Returns:
        Tuple[np.ndarray, np.ndarray]: the possible next states and the rewards for taking any action in the state
    """
    raise NotImplementedError