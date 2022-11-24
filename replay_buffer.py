import random


class Replay_buffer:
  def __init__(self, size: int = 1000):
    """initialize an empty replay buffer with the given size

    Args:
        buffer_size (int, optional): number of items that can be stored in the buffer. Defaults to 1000.
    """
    self.buffer_size: int = size
    self.buffer_list: list = [None] * self.buffer_size
    self._buffer_filled: bool = False
    self._next_item_index: int = 0


  def add_items(self, items: list):
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
    # if buffer is not full, add items to the end of the buffer
    # if self.number_of_items < self.buffer_size - n_new_items:
    if self._next_item_index + n_new_items < self.buffer_size:
      self.buffer_list[self._next_item_index:self._next_item_index + n_new_items] = items
      self._next_item_index = (self._next_item_index + n_new_items) % self.buffer_size
      if self._next_item_index == self.buffer_size:
        self._buffer_filled = True
    else: # if buffer is full, add items to the end of the buffer and overwrite the beginning
      n_items_in_first_batch = self.buffer_size - self._next_item_index
      self.buffer_list[self._next_item_index:self.buffer_size] = items[:n_items_in_first_batch]
      self.buffer_list[:n_new_items - n_items_in_first_batch] = items[n_items_in_first_batch:]
      self._next_item_index = n_new_items - n_items_in_first_batch
      self._buffer_filled = True


  def add_item(self, item):
    """
    add a single item to the replay buffer

    Args:
        item (any): item to add to the buffer
    """
    self.buffer_list[self._next_item_index] = item
    self._next_item_index = (self._next_item_index + 1) % self.buffer_size
    if self._next_item_index == 0:
      self._buffer_filled = True


  def sample_single(self) -> object:
    """
    return a single element from the replay buffer

    Returns:
        object: random element from the buffer
    """
    return random.choice(self.buffer_list)


  def sample_batch(self, batch_size: int = 100) -> list:
    """
    return `batch_size` random elements from the rpelay buffer.
    if `batch_size` is larger than the buffer size, the buffer will be returned in random order.
    If buffer is not full, all elements will be returned in random order.

    Args:
        batch_size (int, optional): number of elements to return. Defaults to 100.

    Returns:
        list: list of random elements from the buffer
    """
    # sample without chooing None values
    if self._buffer_filled:
      return random.sample(self.buffer_list, max(batch_size, self.buffer_size))
    if self._next_item_index < batch_size:
      return random.sample(self.buffer_list[:self._next_item_index], self._next_item_index)
    return random.sample(self.buffer_list[:self._next_item_index], batch_size)


  def __str__(self) -> str:
    return str(self.buffer_list)


  def __len__(self) -> int:
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
    B.add_items(L)
    print(B, len(B))
