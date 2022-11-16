"""
this module implements a framework to solve several problems using Q-learning.
"""
import numpy as np
import tensorflow.keras as keras

from q_problem_interface import Q_learning_problem

class Q_learning_framework:
  def __init__(self, problem: Q_learning_problem, max_episodes: int, learning_rate: float, exploration_rate: float):
    self.problem: Q_learning_problem = problem
    self.max_episodes: int = max_episodes
    self.learning_rate: float = learning_rate
    self.exploration_rate: float = exploration_rate

  def train_model(self, neural_net: keras.Model):
    """
    train a given neural network using Q-learning with given the parameters.

    Args:
        neural_net (keras.Model): a neural network to be trained. This should be a map from state to Q-values for each action (S -> A).
    """
    for episode in range(self.max_episodes):
      # play a single episode
      self.play_episode(neural_net)
      # update the network

  def play_episode(self, neural_net: keras.Model, max_episode_length: int):
    """
    Play a single episode of the bitflip game. The episode ends when the goal is reached or the maximum episode length is reached.
    """
    state = self.problem.gen_start_state()
    for i in range(max_episode_length):
      # choose an action
      action = self.choose_action(state, neural_net)
      # take the action and observe the reward
      new_state, reward = self.problem.take_action(state, action)
      # update the network
      raise NotImplementedError

  def choose_action(self,
        state: np.ndarray,
        neural_net: keras.Model) -> int:
    """
    choose an action using the epsilon-greedy policy. With probability `self.exploration_rate`, choose a random action. Otherwise, choose the action with the highest Q-value.

    Args:
        state (np.ndarray): the current state of the environment
        neural_net (keras.Model): the neural network used to estimate the Q-values

    Returns:
        int: the chosen action
    """
    if np.random.random() < self.exploration_rate:
      return np.random.randint(0, self.problem_size)
    else:
      return np.argmax(neural_net(state))

  