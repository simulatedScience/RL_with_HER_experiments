"""
This implements the bitflip problem using the Q-leaerning interface.
"""
from typing import Tuple

import numpy as np
from q_problem_interface import Q_learning_problem

class Bitflip_problem(Q_learning_problem):
  def __init__(self, problem_size: int):
    self.problem_size: int = problem_size
    self.goal: np.ndarray = self.gen_goal_state()

  def gen_start_state(self) -> np.ndarray:
    """
    generate a random 0-1-array as a start state for the bitflip problem.

    Returns:
        np.ndarray: the start state
    """
    return np.random.randint(0, 2, self.problem_siz, dtype=np.int8)

  def gen_goal_state(self) -> np.ndarray:
    """
    generate a random 0-1-array as a goal state for the bitflip problem.

    Returns:
        np.ndarray: the goal state
    """
    return np.random.randint(0, 2, self.problem_size, dtype=np.int8)

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
    return 1 if np.all(state[action] == self.goal[action]) else 0

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
    new_state = state.copy()
    # flip the bit at position `action`
    new_state[action] = 1 - new_state[action]
    return new_state

  def is_goal(self, state: np.ndarray) -> bool:
    """
    check if the given state is the goal state.

    Args:
        state (np.ndarray): the state to check

    Returns:
        bool: True if the state is the goal state, False otherwise
    """
    return np.all(state == self.goal)

  def get_actions(self, state: np.ndarray) -> np.ndarray:
    """
    get the possible actions for a given state. These are always all integers in range [0, problem_size-1].

    Args:
        state (np.ndarray): the state to get the actions for

    Returns:
        np.ndarray: the actions
    """
    return np.arange(self.problem_size)

  def take_action(self, state: np.ndarray, action: int) -> Tuple(float, np.ndarray):
    """
    take an action in a given state and return the reward and the next state.

    Args:
        state (np.ndarray): the current state
        action (int): the action to take

    Returns:
        Tuple(float, np.ndarray): the reward and the next state
    """
    reward = self.get_reward(state, action)
    next_state = self.get_next_state(state, action)
    return reward, next_state
