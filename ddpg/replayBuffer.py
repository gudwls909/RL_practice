import random
from collections import deque

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.buffer_pointer = 0

    def get_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def add(self, s, a, r, s_):
        if self.buffer_pointer < self.buffer_size:
            self.buffer.append((s, a, r, s_))
            self.buffer_pointer += 1
        else:
            self.buffer.popleft()
            self.buffer.append((s, a, r, s_))

    def count(self):
        return self.buffer_pointer
