"""
visualises the q learning training process and evaluation results
"""

from typing import List

import numpy as np
import matplotlib.pyplot as plt

def plot_success_rate(
      success_rate: List[bool],
      moving_average_window: int = 1,
      show: bool = True,):
  """
  plots the success rate of the neural network during training using a moving average.
  Plot starts after `moving_average_window` episodes-
  
  Args:
      success_rate (List[bool]): list of success rates
      moving_average_window (int, optional): window size for moving average. Defaults to 1.
      show (bool, optional): whether to show the plot. Defaults to True.
  """
  # calculate moving average
  success_rate_moving_average: List[float] = []
  for i in range(moving_average_window, len(success_rate)):
    success_rate_moving_average.append(
        np.mean(success_rate[i - moving_average_window:i + 1]))
  # plot success rate
  x_values: List[int] = list(range(moving_average_window, len(success_rate)))
  plt.plot(x_values, success_rate_moving_average, label="success rate", color="#5588ff")
  plt.xlabel("Episode")
  plt.ylabel("Success rate")
  plt.title("Success rate during training")
  plt.legend()
  if show:
    plt.show()