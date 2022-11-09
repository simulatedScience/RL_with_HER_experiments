import numpy as np
import tensorflow.keras as keras

def gen_bitflip_start(n: int=20) -> np.ndarray:
  return np.random.randint(low=0, high=2, size=n, dtype=np.int8)

def train_model(
  neural_net: keras.Model,
  max_epochs: int,
  learning_rate: float,
  exploration_rate: float):
  play_epoch()

def play_epoch(max_epoch_length: int):
  """
  play a single epoch
  """
