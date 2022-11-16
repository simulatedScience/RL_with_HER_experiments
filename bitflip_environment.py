"""
This implements the bitflip problem using the Q-leaerning interface.
"""
import numpy as np
from q_problem_interface import Q_learning_problem

class Bitflip_problem(Q_learning_problem):
  def __init__(self, problem_size: int):
    self.problem_size: int = problem_size
    self.goal: np.ndarray = self.gen_goal_state()

  def gen_start_state(self) -> np.ndarray:
    """
    generate a random start state for the problem.

    Returns:
        np.ndarray: the start state
    """
    return np.random.randint(0, 2, self.problem_siz, dtype=np.int8)

  def gen_goal_state(self) -> np.ndarray:
    """
    generate a random goal state for the problem.

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
