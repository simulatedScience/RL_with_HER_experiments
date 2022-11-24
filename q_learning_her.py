"""
this module implements a framework to solve several problems using Q-learning using temporal difference learning, a neural network, a replay buffer and hindsight experience replay.

See https://arxiv.org/abs/1707.01495 for more information on hindsight experience replay.
"""
from typing import List, Tuple

import numpy as np
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
      goal: np.ndarray = self.problem.gen_goal_state()
    else: # use the fixed goal
      goal: np.ndarray = self.problem.goal
    # generate a random start state
    state: np.ndarray = self.problem.gen_start_state()

    state_action_history: List[np.ndarray] = [] # list of visited states, actions and rewards
    # play episode
    for i in range(max_episode_length):
      extended_state = self.__get_nn_input(state, goal)
      action = self.choose_action(extended_state, neural_net, self.exploration_rate)
      # take the action and observe the reward and new state
      reward, new_state = self.problem.take_action(state, action)
      state_action_history.append((state, action, reward))
      # add the transition to the replay buffer
      self.replay_buffer.add_item(self.__get_buffer_transition(state, action, reward, new_state, goal))
      if self.problem.is_goal(new_state):
        goal_reached = True
        if self.verbosity > 0:
          print(f"Episode {episode_index} finished successfully after {i+1} steps.")
        break
      state = new_state # update state
    else: # episode ended without reaching the goal
      goal_reached = False
      if self.verbosity > 0:
        print(f"Episode {episode_index} ended without reaching the goal.")
    # add hindsight experience replay samples
    if self.n_her_samples > 0 and i > 0:
      self.__add_her_samples(goal, goal_reached, state_action_history)
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
    # sample a batch from the replay buffer
    batch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
    # get the states, actions, rewards, and new states from the batch
    states, actions, rewards, new_states = self.__get_batch_data(batch)
    # calculate the target Q-values
    target_q_values = self.__get_target_q_values(neural_net, states, actions, rewards, new_states)
    # update the neural network
    neural_net.fit(states, target_q_values, epochs=1, verbose=self.verbosity)


  def __get_batch_data(self, batch_samples: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    for i, sample in enumerate(batch_samples):
      states[i] = sample[0]
      actions[i] = sample[1]
      rewards[i] = sample[2]
      new_states[i] = sample[3]
    return states, actions, rewards, new_states


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
        np.ndarray: the target Q-values. This is a 2D array with shape (batch_size, num_actions) where all the Q-values are the same except for the Q-value for the action taken in each transition.
    """
    batch_size = states.shape[0]
    # get the Q-values for the new states
    next_q_values = neural_net.predict(new_states, verbose=self.verbosity)
    # get the Q-values for the states
    current_q_values = neural_net.predict(states, verbose=self.verbosity)
    # get the Q-values for the actions taken
    int_actions = actions.astype(np.int16)
    q_values_for_actions = current_q_values[np.arange(batch_size), int_actions.reshape(-1)]
    # get the maximum Q-value for each follow-up state in the same shape as the current Q-values
    max_next_q_values = np.max(next_q_values, axis=1)
    # calculate the target Q-values
    target_q_values = q_values_for_actions + self.learning_rate \
        * (rewards + self.discount_factor * max_next_q_values - q_values_for_actions)
    # reshape the target Q-values to be a 2D array with shape (batch_size, num_actions)
    # target_q_values = target_q_values.reshape(-1, 1)
    # set the target Q-values for the actions taken to be the target Q-values calculated above
    current_q_values[np.arange(batch_size), int_actions] = target_q_values
    return current_q_values


  def __get_buffer_transition(self,
        state: np.ndarray,
        action: int,
        reward: float,
        new_state: np.ndarray,
        goal: np.ndarray) -> Tuple[np.ndarray, int, float, np.ndarray, np.ndarray]:
    """
    get a transition for the replay buffer

    Args:
        state (np.ndarray): the state
        action (int): the action
        reward (float): the reward
        new_state (np.ndarray): the new state
        goal (np.ndarray): the goal

    Returns:
        Tuple[np.ndarray, int, float, np.ndarray, np.ndarray]: the transition
    """
    return self.__get_nn_input(state, goal), action, reward, self.__get_nn_input(new_state, goal)


  def __add_her_samples(self,
        goal: np.ndarray,
        goal_reached: bool,
        state_action_history: List[Tuple[np.ndarray, int]]):
    """
    add hindsight experience replay samples to the replay buffer.

    Args:
        goal (np.ndarray): the goal state
        goal_reached (bool): True if the goal was reached, False otherwise
        state_action_history (List[Tuple[np.ndarray, int]]): list of visited states and actions
    """
    for _ in range(self.n_her_samples):
      # generate a random index
      state_index: int = np.random.randint(0, len(state_action_history)-1)
      # get random goal index after the state index
      goal_index: int = np.random.randint(state_index+1, len(state_action_history))
      # get information to add to the replay buffer
      state, action, reward = state_action_history[state_index]
      new_state, *_ = state_action_history[state_index+1]
      new_goal, *_ = state_action_history[goal_index]
      # define input for the replay buffer
      self.replay_buffer.add_item(
          self.__get_buffer_transition(state, action, reward, new_state, new_goal))


  def __get_nn_input(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """
    get the input for the neural network for HER by concatenating the state and the goal.

    Args:
        state (np.ndarray): the current state
        goal (np.ndarray): the goal state

    Returns:
        np.ndarray: the input for the neural network
    """
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
        reward, new_state = self.problem.take_action(state, action)
        if self.problem.is_goal(new_state):
          success_count += 1
          break
        state = new_state # update the state
      else:
        # if the episode ends without reaching the goal, print the action sequence
        print(start_state, action_sequence)
    return success_count / num_episodes