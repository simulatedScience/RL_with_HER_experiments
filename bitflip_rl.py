"""
solves the bitflip problem using reinforcement learning with temporal difference learning.
"""
from typing import Tuple

import numpy as np
import tensorflow.keras as keras

from general_q_learning import Q_learning_framework
from q_learning_her import Q_learning_framework_her
from bitflip_environment import Bitflip_problem
from q_problem_interface import Q_learning_problem
from q_learning_visualisations import plot_success_rate

def define_model(
      input_shape: Tuple[int],
      problem: Q_learning_problem) -> keras.Model:
  # define the neural network using keras functional API
  # this should be a map from state to Q-values for each action (S -> A)
  if len(input_shape) == 1:
    input_shape = input_shape[0]
  inputs: keras.layers.Input = keras.layers.Input(shape=input_shape)
  x: keras.layers.Dense = keras.layers.Dense(256, activation="relu")(inputs)
  # x: keras.layers.Dense = keras.layers.Dense(np.sum(problem.get_state_size()), activation="relu")(inputs)
  outputs: keras.layers.Dense = keras.layers.Dense(problem.get_num_actions(), activation="linear")(x)
  neural_net: keras.Model = keras.Model(inputs=inputs, outputs=outputs)
  neural_net.compile(optimizer="adam", loss="mse")
  return neural_net

def define_q_trainer(
      trainer: Q_learning_framework,
      problem: Q_learning_problem,
      max_episode_length: int,
      learning_rate: float,
      exploration_rate: float,
      discount_factor: float,
      batch_size: int,
      replay_buffer_size: int,
      n_her_samples: int = 4,
      fixed_goal: bool = False,
      verbosity: int = 0) -> Q_learning_framework:
      if trainer == Q_learning_framework_her:
        return Q_learning_framework_her(
            problem=problem,
            max_episode_length=max_episode_length,
            learning_rate=learning_rate,
            exploration_rate=exploration_rate,
            discount_factor=discount_factor,
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            n_her_samples=n_her_samples,
            fixed_goal=fixed_goal,
            verbosity=verbosity)
      return Q_learning_framework(
            problem=problem,
            max_episode_length=max_episode_length,
            learning_rate=learning_rate,
            exploration_rate=exploration_rate,
            discount_factor=discount_factor,
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            verbosity=verbosity)

if __name__ == "__main__":
  # problem parameters
  problem_size: int = 5
  # parameters for training
  max_episode_length: int = int(1.5*problem_size)
  learning_rate: float = 0.001
  exploration_rate: float = 0.2
  discount_factor: float = 0.9
  batch_size: int = 32
  replay_buffer_size: int = 128
  train_max_episodes: int = 10000
  train_max_time_s: int = 60 # 30 minutes
  # parameters for hindsight experience replay
  trainer = "her" # "her" or "q"
  n_her_samples: int = 2
  fixed_goal: bool = False
  # parameters for NN evaluation
  eval_num_episodes: int = 100 # number of episodes to evaluate over
  eval_episode_length: int = max_episode_length
  verbosity: int = 0


  # define instance of Q-learning problem
  problem: Bitflip_problem = Bitflip_problem(problem_size)
  # define neural network
  if trainer == "her":
    framework = Q_learning_framework_her
    input_shape: Tuple[int] = (2*problem_size,)
    neural_net: keras.Model = define_model(input_shape, problem)
  else:
    framework = Q_learning_framework
    input_shape: Tuple[int] = problem.get_state_size()
    neural_net: keras.Model = define_model(input_shape, problem)

  # define the Q-learning framework
  q_learning_framework: Q_learning_framework = define_q_trainer(
      trainer=framework,
      problem=problem,
      max_episode_length=max_episode_length,
      learning_rate=learning_rate,
      exploration_rate=exploration_rate,
      discount_factor=discount_factor,
      batch_size=batch_size,
      replay_buffer_size=replay_buffer_size,
      n_her_samples=n_her_samples,
      fixed_goal=fixed_goal,
      verbosity=verbosity)

  # train the neural network
  # perform runtime evaluation using cProfile
  # import cProfile
  # import pstats
  # profile = cProfile.Profile()
  # success_rate = profile.runcall(q_learning_framework.train_model,
  #   neural_net=neural_net,
  #     max_episode_length=max_episode_length,
  #     max_episodes=train_max_episodes,
  #     max_time_s=train_max_time_s)
  # # print the results of the cProfile
  # stats = pstats.Stats(profile)
  # stats.sort_stats("cumtime")
  # stats.print_stats(30)
  success_rate = q_learning_framework.train_model(
      neural_net=neural_net,
      max_episode_length=max_episode_length,
      max_episodes=train_max_episodes,
      max_time_s=train_max_time_s)

  # # plot the success rate
  # plot_success_rate(
  #     success_rate=success_rate,
  #     moving_average_window=64,
  #     show=True)

  # # evaluate the neural network
  # success_rate = q_learning_framework.evaluate_model(neural_net=neural_net,
  #     num_episodes=eval_num_episodes,
  #     max_episode_length=eval_episode_length)
  # print(f"Success rate: {success_rate}")
