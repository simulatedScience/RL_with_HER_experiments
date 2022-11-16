"""
this module implements a framework to solve several problems using Q-learning.
"""
import numpy as np
import tensorflow.keras as keras

from replay_buffer import Replay_buffer
from q_problem_interface import Q_learning_problem

class Q_learning_framework:
  def __init__(self,
      problem: Q_learning_problem,
      max_episodes: int,
      learning_rate: float,
      exploration_rate: float,
      buffer_size: int = 1000):
    """
    initialize the Q-learning framework.

    Args:
        problem (Q_learning_problem): the problem to solve
        max_episodes (int): the maximum number of episodes to play
        learning_rate (float): the learning rate for the neural network
        exploration_rate (float): the probability of choosing a random action
    """
    self.problem: Q_learning_problem = problem
    self.max_episodes: int = max_episodes
    self.learning_rate: float = learning_rate
    self.exploration_rate: float = exploration_rate
    # create the replay buffer
    # buffer should never be smaller than episode length
    self.buffer_size: int = max(buffer_size, max_episodes)
    self.replay_buffer: Replay_buffer = Replay_buffer(size=self.buffer_size)


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
      reward, new_state = self.problem.take_action(state, action)
      # add the transition to the replay buffer
      self.replay_buffer.add_transition(state, action, reward, new_state)


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