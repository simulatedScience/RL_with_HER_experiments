import random


class Replay_Buffer():
  def __init__(self, size: int = 1000):
    """initialize an empty replay buffer with the given size

    Args:
        buffer_size (int, optional): number of items that can be stored in the buffer. Defaults to 1000.
    """
    self.buffer_size: int = size
    self.buffer_list: list = [None] * self.buffer_size
    self._next_item_index: int = 0


  def add_items_to_buffer(self, items: list):
    """add a number of items to the replay buffer

    Args:
        items (list): list of items to add to the buffer. The list should never include more than `buffer_size` elements.

    Raises:
        ValueError: If number of new items exceeds size of the buffer. If you can't increase the buffer size, split the pushed items.
    """
    n_new_items = len(items)
    if n_new_items > self.buffer_size:
      raise ValueError(
          f"Too many items pushed into buffer. Tried adding {n_new_items} items to buffer of size {self.buffer_size}.")
    # if self.number_of_items < self.buffer_size - n_new_items:
    if self._next_item_index + n_new_items < self.buffer_size:
      self.buffer_list[self._next_item_index:self._next_item_index + n_new_items] = items
      self._next_item_index += n_new_items
    else:
      n_items_in_first_batch = self.buffer_size - self._next_item_index
      self.buffer_list[self._next_item_index:self.buffer_size] = items[:n_items_in_first_batch]
      self.buffer_list[:n_new_items - n_items_in_first_batch] = items[n_items_in_first_batch:]
      self._next_item_index = n_new_items - n_items_in_first_batch


  def add_item_to_buffer(self, item):
    """add a single item to the replay buffer

    Args:
        item ([type]): [description]
    """
    self.buffer_list[self._next_item_index] = item
    self._next_item_index = (self._next_item_index + 1) % self.buffer_size


  def single_sample(self):
    """return a single element from the replay buffer

    Returns:
        [type]: [description]
    """
    return random.choice(self.buffer_list)


  def sample_batch(self, batch_size: int = 100):
    """return `batch_size` random elements from the rpelay buffer.

    Args:
        batch_size (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    return random.choices(self.buffer_list, k=batch_size)


  def __str__(self):
    return str(self.buffer_list)


  def __len__(self):
    return len(self.buffer_list)


if __name__ == "__main__":
  # test
  B = Replay_Buffer(size=20)

  test_list = (5, 10, 7, 15, 10, 5, 12, 20)
  # test_list = (20, 10, 8, 2, 1)

  prev_n = 0
  for n in test_list:
    L = list(range(prev_n, prev_n + n))
    prev_n += n
    B.add_items_to_buffer(L)
    print(B, len(B))