import random
from collections import deque, namedtuple
import gym
class ReplayMemory():
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition',(('state', 'action', 'reward', 'state_', 'is_terminal')))
    
    def __len__(self):
        return len(self.memory)

    def push_pop(self, *args):
        self.memory.append(self.Transition(*args))
    
    def unzip(self, transitions):
        return self.Transition(*zip(*transitions))

    def sample(self, batch_size):
        data = random.sample(self.memory, batch_size)
        return self.unzip(data)

    def trajectory(self, batch_size):
        data = self.memory[-batch_size:]
        return self.unzip(data)

def action_undiscre(env, output_dim, action_int):
    assert  action_int >= 0 and action_int <= (output_dim - 1)
    discre = (env.action_space.high - env.action_space.low) / (output_dim - 1)
    action_float = action_int * discre + env.action_space.low
    return action_float
