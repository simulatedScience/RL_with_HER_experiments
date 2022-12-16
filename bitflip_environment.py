"""
This implements the bitflip problem using the Q-leaerning interface.
"""
from typing import Tuple

import numpy as np
from q_problem_interface import Q_learning_problem

class Bitflip_problem(Q_learning_problem):
  def __init__(self, problem_size: int):
    self.problem_size: int = problem_size
    self.goal: np.ndarray = np.ones(problem_size, dtype=np.int8)

  def gen_start_state(self) -> np.ndarray:
    """
    generate a random 0-1-array as a start state for the bitflip problem.

    Returns:
        np.ndarray: the start state
    """
    return np.random.randint(0, 2, self.problem_size, dtype=np.int8)

  def gen_goal_state(self) -> np.ndarray:
    """
    generate a random 0-1-array as a goal state for the bitflip problem.
    This goal is saved in the `goal` attribute.

    Returns:
        np.ndarray: the goal state
    """
    self.goal = np.random.randint(0, 2, self.problem_size, dtype=np.int8)
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
        state: np.ndarray,
        action: int,
        goal: np.ndarray = None) -> Tuple[float, bool]:
    """
    get the reward for taking an action in a given state.

    Args:
        state (np.ndarray): the current state
        action (int): the action to take
        goal (np.ndarray, optional): the goal state. If goal is None, the goal state stored in the `goal` attribute is used. Defaults to None.

    Returns:
        float: the reward for taking the action
        bool: True if the goal state is reached, False otherwise
    """
    if goal is None:
      goal = self.goal
    reward = 1 if np.all(state[action] == goal[action]) else 0
    goal_reached = np.all(state == goal)
    return reward, goal_reached

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
    check if the given state is the goal state stored in the `goal` attribute.

    Args:
        state (np.ndarray): the state to check

    Returns:
        bool: True if the state is the goal state, False otherwise

    Raises:
        ValueError: if the shape of the state does not match the shape of the goal state
    """
    if state.shape != self.goal.shape:
      raise ValueError(f"state shape {state.shape} does not match goal shape {self.goal.shape}")
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

  def take_action(self, state: np.ndarray, action: int) -> Tuple[float, np.ndarray, bool]:
    """
    take an action in a given state and return the reward and the next state.

    Args:
        state (np.ndarray): the current state
        action (int): the action to take

    Returns:
        float: the reward for taking the action
        np.ndarray: the next state after taking the action
        bool: True if the goal state is reached, False otherwise
    """
    reward, goal_reached = self.get_reward(state, action, self.goal)
    next_state = self.get_next_state(state, action)
    return reward, next_state, goal_reached

if __name__ == "__main__":
  # test the bitflip problem
  problem = Bitflip_problem(4)
  print(problem.gen_start_state())
  print(problem.gen_goal_state())
  print(problem.get_reward(np.array([0, 1, 0, 1]), 1))
  print(problem.get_next_state(np.array([0, 1, 0, 1]), 1))
  print(problem.is_goal(np.array([0, 1, 0, 1])))
  print(problem.get_actions(np.array([0, 1, 0, 1])))
  print(problem.take_action(np.array([0, 1, 0, 1]), 1))
