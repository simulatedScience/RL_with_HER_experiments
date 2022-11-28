"""
this module implements a framework to solve several problems using Q-learning using temporal difference learning, a neural network, a replay buffer and hindsight experience replay.

See https://arxiv.org/abs/1707.01495 for more information on hindsight experience replay.
"""
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from general_q_learning import Q_learning_framework
from q_problem_interface import Q_learning_problem

class Q_learning_framework_her(Q_learning_framework):
  """
  implements a framework to solve several problems using Q-learning using temporal difference learning, a neural network, a replay buffer and hindsight experience replay with the future strategy for goal generation.

  This extends the Q_learning_framework class.
  """
  def __init__(
        self,
        problem: Q_learning_problem,
        max_episode_length: int,
        learning_rate: float,
        exploration_rate: float,
        discount_factor: float,
        batch_size: int,
        replay_buffer_size: int,
        n_her_samples: int = 4,
        fixed_goal: bool = False,
        verbosity: int = 0):
    """
    initialises the Q-learning framework.

    Args:
        problem (Q_learning_problem): instance of the problem to solve
        max_episode_length (int): maximum length of an episode
        learning_rate (float): learning rate for the neural network
        exploration_rate (float): exploration rate for the neural network
        discount_factor (float): discount factor for the temporal difference learning
        batch_size (int): batch size for the neural network
        replay_buffer_size (int): size of the replay buffer
        n_her_samples (float, optional): number of hindsight experience replay samples. Defaults to 4.
        verbosity (int, optional): verbosity level. Defaults to 0.
    """
    super().__init__(problem=problem,
        max_episode_length=max_episode_length,
        learning_rate=learning_rate,
        exploration_rate=exploration_rate,
        discount_factor=discount_factor,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        verbosity=verbosity)
    self.n_her_samples: float = n_her_samples
    self.fixed_goal: bool = fixed_goal
    # determine input shape of neural network
    sample_state: np.ndarray = self.__get_nn_input(
        self.problem.gen_start_state(), self.problem.get_goal())
    self.nn_input_shape: Tuple[int] = sample_state.shape


  def play_episode(self,
        neural_net: keras.Model,
        max_episode_length: int,
        episode_index: int) -> bool:
    """
    Play a single episode of the bitflip game. The episode ends when the goal is reached or the maximum episode length is reached.
    Before the episode starts, the goal is randomly generated.
    After the episode ends, the episode is added to the replay buffer with extra samples generated using hindsight experience replay.

    Args:
        neural_net (keras.Model): the neural network used to estimate the Q-values. Input shape: (2*state_size,), output shape: (action_size,)
        max_episode_length (int): the maximum number of steps to take in an episode
        episode_index (int): the index of the episode

    Returns:
        bool: True if the goal was reached, False otherwise
    """
    if not self.fixed_goal: # generate a random goal
      goal: tf.Tensor = self.problem.gen_goal_state()
    else: # use the fixed goal
      goal: tf.Tensor = self.problem.goal
    # generate a random start state
    state: tf.Tensor = self.problem.gen_start_state()

    state_action_history: list = [] # list of visited states, actions and rewards
    # play episode
    for i in range(max_episode_length):
      extended_state = self.__get_nn_input(state, goal)
      action = self.choose_action(extended_state, neural_net, self.exploration_rate)
      # take the action and observe the reward and new state
      reward, new_state, goal_reached = self.problem.take_action(state, action)
      state_action_history.append((state, action, reward))
      # add the transition to the replay buffer
      self.replay_buffer.add_item(
          self.__get_buffer_transition(state, action, reward, new_state, goal, goal_reached))
      if goal_reached:
        if self.verbosity > 0:
          print(f"Episode {episode_index} finished successfully after {i+1} steps.")
        break
      state = new_state # update state
    else: # episode ended without reaching the goal
      if self.verbosity > 0:
        print(f"Episode {episode_index} ended without reaching the goal.")
    # add hindsight experience replay samples
    if self.n_her_samples > 0 and i > 0:
      self.__add_her_samples(goal, state_action_history)
    # update the network
    self.update_network(neural_net)
    return goal_reached


  def update_network(self, neural_net: keras.Model):
    """
    update the neural network using the replay buffer and the temporal difference update rule.
    Train the network for a single epoch on a batch of transitions randomly sampled from the replay buffer.

    Args:
        neural_net (keras.Model): the neural network to be updated
    """
    super().update_network(neural_net)


  def __get_buffer_transition(self,
        state: tf.Tensor,
        action: int,
        reward: float,
        new_state: tf.Tensor,
        goal: tf.Tensor,
        goal_reached: bool) -> Tuple[tf.Tensor, int, float, tf.Tensor, tf.Tensor, bool]:
    """
    get a transition for the replay buffer

    Args:
        state (tf.Tensor): the state
        action (int): the action
        reward (float): the reward
        new_state (tf.Tensor): the new state
        goal (tf.Tensor): the goal
        goal_reached (bool): True if the goal was reached, False otherwise

    Returns:
        Tuple[tf.Tensor, int, float, tf.Tensor, tf.Tensor, bool]: the transition for the replay buffer (state, action, reward, new_state, goal, goal_reached)
    """
    return self.__get_nn_input(state, goal), action, reward, self.__get_nn_input(new_state, goal), goal_reached


  def __add_her_samples(self,
        goal: tf.Tensor,
        state_action_history: list):
    """
    add hindsight experience replay samples to the replay buffer.

    Args:
        goal (tf.Tensor): the goal state
        state_action_history (list): list of visited states and actions
    """
    for _ in range(self.n_her_samples):
      # choose a random state from the episode
      state_index: int = np.random.randint(0, len(state_action_history)-1)
      # get random goal index after the state index (future strategy)
      goal_index: int = np.random.randint(state_index+1, len(state_action_history))
      # get information to add to the replay buffer
      state, action, reward = state_action_history[state_index]
      new_state, *_ = state_action_history[state_index+1]
      # get reward considering the new goal
      reward, goal_reached = self.problem.get_reward(new_state, action, goal)
      new_goal, *_ = state_action_history[goal_index]
      # define input for the replay buffer
      self.replay_buffer.add_item(
          self.__get_buffer_transition(state, action, reward, new_state, new_goal, goal_reached))


  def __get_nn_input(self, state: tf.Tensor, goal: tf.Tensor) -> tf.Tensor:
    """
    get the input for the neural network for HER by concatenating the state and the goal.

    Args:
        state (tf.Tensor): the current state
        goal (tf.Tensor): the goal state

    Returns:
        np.ndarray: the input for the neural network
    """
    # return tf.concat(state, goal)
    return np.concatenate((state, goal))

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
    goal = self.problem.get_goal()
    for i in range(num_episodes):
      start_state = self.problem.gen_start_state()
      # generate a random goal
      if not self.fixed_goal:
        goal = self.problem.gen_goal_state()
      state = start_state
      action_sequence = []
      for j in range(max_episode_length):
        extended_state = self.__get_nn_input(state, goal)
        action = self.choose_action(extended_state, neural_net, exploration_rate=0.2)
        action_sequence.append(action)
        reward, new_state, goal_reached = self.problem.take_action(state, action)
        if goal_reached:
          success_count += 1
          break
        state = new_state # update the state
      else:
        # if the episode ends without reaching the goal, print the action sequence
        print(f"eval her, loss: {start_state.numpy()} -> {action_sequence} -> {new_state.numpy()}")
    return success_count / num_episodes