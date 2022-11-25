"""
this module implements a framework to solve several problems using Q-learning using temporal difference learning, a neural network and a replay buffer.
"""
import time
from typing import Tuple, List

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
      replay_buffer_size: int = 1024,
      verbosity: int = 0):
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
        verbosity (int): the verbosity level for learning. 0 is silent, 1 prints some information about episode success and 2 also prints information about the neural network training.
    """
    self.problem: Q_learning_problem = problem
    self.max_episode_length: int = max_episode_length
    self.learning_rate: float = learning_rate
    self.exploration_rate: float = exploration_rate
    self.discount_factor: float = discount_factor
    self.batch_size: int = batch_size
    self.verbosity: int = verbosity
    # create the replay buffer
    # buffer should never be smaller than episode length
    if replay_buffer_size < max_episode_length:
      print(f"Warning: replay buffer size is smaller than episode length. Setting replay buffer size to {max_episode_length}.")
      replay_buffer_size = max_episode_length
    self.BUFFER_SIZE: int = replay_buffer_size
    self.replay_buffer: Replay_buffer = Replay_buffer(size=self.BUFFER_SIZE)
    # set input shape of neural network
    self.nn_input_shape: Tuple[int] = self.problem.get_state_size()


  def train_model(self,
        neural_net: keras.Model,
        max_episode_length: int,
        max_episodes: int = 1000,
        max_time_s: int = 300) -> List[bool]:
    """
    train a given neural network using Q-learning with given the parameters.
    For each episode, record whether the goal was reached or not.

    Args:
        neural_net (keras.Model): a neural network to be trained. This should be a map from state to Q-values for each action (S -> A).
        max_episode_length (int): the maximum number of steps to take in an episode
        max_episodes (int, optional): the maximum number of episodes to play
        max_time_s (int, optional): the maximum number of seconds to train for

    Returns:
        list: a list of booleans indicating whether the goal was reached in each episode
    """
    # initialize the list of goals reached
    goals_reached = []
    episode_index = 0
    status_print_count = 10
    print_counter = 1
    last_print_end_index = 0
    start_time = time.perf_counter()
    while episode_index < max_episodes and time.perf_counter() - start_time < max_time_s:
      # play a single episode
      success = self.play_episode(neural_net, max_episode_length, episode_index)
      goals_reached.append(success)
      # print success rate every 10% of time
      if time.perf_counter() - start_time >print_counter * (max_time_s / status_print_count):
        print(f"Success rate after {episode_index} episodes: {np.mean(goals_reached[last_print_end_index:episode_index]):.2f}")
        last_print_end_index = episode_index
        print_counter += 1
      episode_index += 1
    print(f"Finished training after {episode_index} episodes and {time.perf_counter() - start_time} seconds.")
    return goals_reached


  def play_episode(self,
        neural_net: keras.Model,
        max_episode_length: int,
        episode_index: int) -> bool:
    """
    Play a single episode of the bitflip game. The episode ends when the goal is reached or the maximum episode length is reached.

    Args:
        neural_net (keras.Model): the neural network used to estimate the Q-values
        max_episode_length (int): the maximum number of steps to take in an episode
        episode_index (int): the index of the episode

    Returns:
        bool: True if the goal was reached, False otherwise
    """
    state = self.problem.gen_start_state()
    for i in range(max_episode_length):
      action = self.choose_action(state, neural_net, self.exploration_rate)
      # take the action and observe the reward and new state
      reward, new_state, goal_reached = self.problem.take_action(state, action)
      # add the transition to the replay buffer
      self.replay_buffer.add_item(self.__get_buffer_transition(state, action, reward, new_state, goal_reached))
      if goal_reached:
        if self.verbosity > 0:
          print(f"Episode {episode_index} finished successfully after {i+1} steps.")
        break
      state = new_state
    else: # episode ended without reaching the goal
      if self.verbosity > 0:
        print(f"Episode {episode_index} ended without reaching the goal.")
    # update the network
    self.update_network(neural_net)
    return goal_reached


  def choose_action(self,
        state: np.ndarray,
        neural_net: keras.Model,
        exploration_rate: float) -> int:
    """
    choose an action using the epsilon-greedy policy. With probability `self.exploration_rate`, choose a random action. Otherwise, choose the action with the highest Q-value.

    Args:
        state (np.ndarray): the current state of the environment
        neural_net (keras.Model): tpredhe neural network used to estimate the Q-values

    Returns:
        int: the chosen action
    """
    if np.random.random() < exploration_rate:
      return np.random.randint(0, self.problem.get_num_actions())
    else:
      predictions = neural_net.predict(state.reshape(1, -1), verbose=self.verbosity)
      return np.argmax(predictions)


  def __get_buffer_transition(self,
        state: np.ndarray,
        action: int,
        reward: float,
        new_state: np.ndarray,
        goal_reached: bool) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
    """
    get a transition for the replay buffer. This is a tuple of (state, action, reward, new_state).

    Args:
        state (np.ndarray): the current state
        action (int): the action taken
        reward (float): the reward received
        new_state (np.ndarray): the new state after taking the action
        goal_reached (bool): whether the goal was reached

    Returns:
        tuple: a tuple of (state, action, reward, new_state, goal_reached) for the replay buffer
    """
    return (state, action, reward, new_state, goal_reached)


  def __get_batch_data(self, batch_samples: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    get the states, actions, rewards, and new states from the replay buffer.

    Args:
        batch_size (int): the number of transitions to sample from the replay buffer

    Returns:
        tuple: a tuple of (states, actions, rewards, new_states)
    """
    batchsize = len(batch_samples)
    # state size is the size of the state plus the size of the goal
    states = np.zeros((batchsize, *self.nn_input_shape))
    actions = np.zeros((batchsize, 1))
    rewards = np.zeros((batchsize, 1))
    new_states = np.zeros((batchsize, *self.nn_input_shape))
    goal_reached = np.zeros(batchsize)
    for i, sample in enumerate(batch_samples):
      states[i] = sample[0]
      actions[i] = sample[1]
      rewards[i] = sample[2]
      new_states[i] = sample[3]
      goal_reached[i] = sample[4]
    return states, actions, rewards, new_states, goal_reached


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
    states, actions, rewards, new_states, goal_reached = self.__get_batch_data(batch)
    # calculate the target Q-values
    target_q_values = self.__get_target_q_values(neural_net, states, actions, rewards, new_states, goal_reached)
    # update the neural network
    neural_net.fit(states, target_q_values, epochs=1, verbose=self.verbosity)


  def __get_target_q_values(self, 
        neural_net: keras.Model,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        new_states: np.ndarray,
        goal_reached: np.ndarray) -> np.ndarray:
    """
    get the target Q-values for the given batch according to the Bellman equation.

    Args:
        neural_net (keras.Model): neural network
        states (np.ndarray): states in the batch
        actions (np.ndarray): actions in the batch
        rewards (np.ndarray): rewards for each action
        new_states (np.ndarray): new states after taking the actions

    Returns:
        np.ndarray: the target Q-values. This is a 2D array with shape (batch_size, num_actions) where all the Q-values are the same except for the Q-value for the action taken in each transition.
    """
    batch_size = states.shape[0]
    # get the Q-values for the new states if the goal was not reached, otherwise, set the Q-values to 0
    next_q_values = np.zeros((batch_size, self.problem.get_num_actions()))
    if 0 in goal_reached:
      predict_states = new_states[goal_reached == 0, :]
      next_q_values[goal_reached == 0, :] = neural_net.predict(predict_states, verbose=self.verbosity)
    # get the Q-values for the states
    current_q_values = neural_net.predict(states, verbose=self.verbosity)
    # get the Q-values for the actions taken
    int_actions = actions.astype(np.int16)
    # get the maximum Q-value for each follow-up state in the same shape as the current Q-values
    max_next_q_values = np.max(next_q_values, axis=1)
    # calculate the target Q-values
    target_q_values = rewards + self.discount_factor * max_next_q_values # TD target
    # set the target Q-values for the actions taken to be the target Q-values calculated above
    current_q_values[np.arange(batch_size), int_actions] = target_q_values
    return current_q_values


  def evaluate_model(self,
        neural_net: keras.Model,
        num_episodes: int,
        max_episode_length: int):
    """
    evaluate the performance of the neural network by playing multiple episodes of the game and measuring the success rate.

    Args:
        neural_net (keras.Model): the neural network to be evaluated
        num_episodes (int): the number of episodes to play
        max_episode_length (int): the maximum number of steps in each episode

    Returns:
        float: the success rate
    """
    success_count = 0
    for i in range(num_episodes):
      start_state = self.problem.gen_start_state()
      state = start_state
      action_sequence = []
      for j in range(max_episode_length):
        action = self.choose_action(state, neural_net, exploration_rate=0.2)
        action_sequence.append(action)
        reward, new_state, goal_reached = self.problem.take_action(state, action)
        if goal_reached:
          success_count += 1
          break
        state = new_state # update the state
      else:
        # if the episode ends without reaching the goal, print the action sequence
        print(start_state, action_sequence)
    return success_count / num_episodes