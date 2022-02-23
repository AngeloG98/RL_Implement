import random
from collections import namedtuple
import gym
class ReplayMemory():
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.memory = []
        self.Transition = namedtuple('Transition',(('state', 'action', 'reward', 'state_', 'is_terminal')))
    
    def __len__(self):
        return len(self.memory)

    def push_pop(self, *args):
        self.memory.append(self.Transition(*args))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
    
    def unzip(self, transitions):
        return self.Transition(*zip(*transitions))

    def sample(self, batch_size):
        data = random.sample(self.memory, batch_size)
        return self.unzip(data)

    def trajectory(self, batch_size):
        data = self.memory[-batch_size:]
        return self.unzip(data)

    
