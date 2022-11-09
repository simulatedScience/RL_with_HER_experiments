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
  max_episodes: int,
  learning_rate: float,
  exploration_rate: float):
  """
  train model with given parameters for a given number of episodes

  Args:
      neural_net (keras.Model): _description_
      max_episodes (int): _description_
      learning_rate (float): _description_
      exploration_rate (float): _description_
  """
  play_episode()

def play_episode(max_episode_length: int):
  """
  play a single episode
  """
