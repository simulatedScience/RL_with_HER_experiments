"""
this module implements a framework to solve several problems using Q-learning.
"""
import time
from typing import Tuple

import numpy as np
import tensorflow.keras as keras

from replay_buffer import Replay_buffer
from q_problem_interface import Q_learning_problem

class Q_learning_framework:
  def __init__(self,
      problem: Q_learning_problem,
      max_episode_length: int,
      learning_rate: float = 0.01,
      exploration_rate: float = 0.1,
      discount_factor: float = 0.9,
      batch_size: int = 32,
      replay_buffer_size: int = 1024):
    """
    initialize the Q-learning framework.

    Args:
        problem (Q_learning_problem): the problem to solve
        max_episode_length (int): the maximum number of steps to take in an episode
        learning_rate (float): the learning rate for the neural network
        exploration_rate (float): the probability of choosing a random action
        discount_factor (float): the discount factor for the temporal difference update rule
        batch_size (int): the number of transitions to sample from the replay buffer
        replay_buffer_size (int): the maximum number of transitions to store in the replay buffer
    """
    self.problem: Q_learning_problem = problem
    self.max_episode_length: int = max_episode_length
    self.learning_rate: float = learning_rate
    self.exploration_rate: float = exploration_rate
    self.discount_factor: float = discount_factor
    self.batch_size: int = batch_size
    # create the replay buffer
    # buffer should never be smaller than episode length
    if replay_buffer_size < max_episode_length:
      print(f"Warning: replay buffer size is smaller than episode length. Setting replay buffer size to {max_episode_length}.")
      replay_buffer_size = max_episode_length
    self.BUFFER_SIZE: int = replay_buffer_size
    self.replay_buffer: Replay_buffer = Replay_buffer(size=self.BUFFER_SIZE)


  def train_model(self,
        neural_net: keras.Model,
        max_episode_length: int,
        max_episodes: int = 1000,
        max_time_s: int = 300):
    """
    train a given neural network using Q-learning with given the parameters.

    Args:
        neural_net (keras.Model): a neural network to be trained. This should be a map from state to Q-values for each action (S -> A).
        max_episode_length (int): the maximum number of steps to take in an episode
        max_episodes (int, optional): the maximum number of episodes to play
        max_time_s (int, optional): the maximum number of seconds to train for
    """
    episode_index = 0
    start_time = time.perf_counter()
    while episode_index < max_episodes and time.perf_counter() - start_time < max_time_s:
      # play a single episode
      self.play_episode(neural_net, max_episode_length, episode_index)
      episode_index += 1
    print(f"Finished training after {episode_index} episodes and {time.perf_counter() - start_time} seconds.")

  def play_episode(self,
        neural_net: keras.Model,
        max_episode_length: int,
        episode_index: int):
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
      self.replay_buffer.add_item(self.__get_buffer_transition(state, action, reward, new_state))
      if self.problem.is_goal_state(new_state):
        break
    else: # episode ended without reaching the goal
      print(f"Episode {episode_index} ended without reaching the goal.")
    # update the network
    self.update_network(neural_net)


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

  def __get_buffer_transition(self,
        state: np.ndarray,
        action: int,
        reward: float,
        new_state: np.ndarray):
    """
    get a transition for the replay buffer. This is a tuple of (state, action, reward, new_state).

    Args:
        state (np.ndarray): the current state
        action (int): the action taken
        reward (float): the reward received
        new_state (np.ndarray): the new state after taking the action
    """
    return (state, action, reward, new_state)


  def __get_batch_data(self, batch_samples: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    get the states, actions, rewards, and new states from the replay buffer.

    Args:
        batch_size (int): the number of transitions to sample from the replay buffer

    Returns:
        tuple: a tuple of (states, actions, rewards, new_states)
    """
    states = np.zeros((len(batch_samples), self.problem.state_size))
    actions = np.zeros((len(batch_samples), 1))
    rewards = np.zeros((len(batch_samples), 1))
    new_states = np.zeros((len(batch_samples), self.problem.state_size))
    for i, sample in enumerate(batch_samples):
      states[i] = sample[0]
      actions[i] = sample[1]
      rewards[i] = sample[2]
      new_states[i] = sample[3]
    return states, actions, rewards, new_states


  def update_network(self, neural_net: keras.Model):
    """
    update the neural network using the replay buffer and the temporal difference update rule.
    Train the network for a single epoch on a batch of transitions randomly sampled from the replay buffer.

    Args:
        neural_net (keras.Model): the neural network to be updated
    """
    # sample a batch from the replay buffer
    batch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
    # get the states, actions, rewards, and new states from the batch
    states, actions, rewards, new_states = self.__get_batch_data(batch)
    # calculate the target Q-values
    target_q_values = self.__get_target_q_values(neural_net, states, actions, rewards, new_states)
    # update the neural network
    neural_net.fit(states, target_q_values, epochs=1, verbose=0)

  def __get_target_q_values(self, 
        neural_net: keras.Model,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        new_states: np.ndarray) -> np.ndarray:
    """
    get the target Q-values for the given batch according to the Bellman equation.

    Args:
        neural_net (keras.Model): neural network
        states (np.ndarray): states in the batch
        actions (np.ndarray): actions in the batch
        rewards (np.ndarray): rewards for each action
        new_states (np.ndarray): new states after taking the actions

    Returns:
        np.ndarray: the target Q-values
    """
    # get the Q-values for the visited states
    current_q_values = neural_net(states)
    # get the Q-values for the follow-up states
    next_q_values = neural_net(new_states)
    max_next_q_values = np.max(next_q_values, axis=1)
    # calculate the updated target Q-values
    target_q_values = current_q_values + self.learning_rate \
        * (rewards + self.discount_factor * max_next_q_values - current_q_values)
    return target_q_values