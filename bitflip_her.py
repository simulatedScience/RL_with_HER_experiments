"""
This module implements a reinforcement learning algorithm to solve the bitflip environment:

1. generate two random tuples of length n with values 0 or 1.
2. one tuple is the initial state, the other the goal.
3. There are n actions, each flipping a single bit in the tuple.
"""
import numpy as np
import tensorflow.keras as keras

def gen_bitflip_start(n: int=20) -> np.ndarray:
  return np.random.randint(low=0, high=2, size=n, dtype=np.int8)

def train_model(
  neural_net: keras.Model,
  problem_size: int,
  max_episodes: int,
  learning_rate: float,
  exploration_rate: float):
  """
  train a given neural network using Q-learning with given the parameters.

  Args:
      neural_net (keras.Model): a neural network to be trained
      problem_size (int): the size of the bitflip problem
      max_episodes (int): number of episodes to train for
      learning_rate (float): learning rate for Q-learning
      exploration_rate (float): exploration rate for epsilon-greedy policy
  """
  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
  for episode in range(max_episodes):
    # play a single episode
    play_episode()
    # update the network


def play_episode(max_episode_length: int):
  """
  Play a single episode of the bitflip game. The episode ends when the goal is reached or the maximum episode length is reached.
  """
  for i in range(max_episode_length):
    # choose an action
    
    # take the action
    # observe the reward
    # observe the next state
    # update the network

