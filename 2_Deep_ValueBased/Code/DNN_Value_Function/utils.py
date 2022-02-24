import random
from collections import deque, namedtuple

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

def reward_func(env, state):
    x, x_dot, theta, theta_dot = state
    r1 = ((env.x_threshold - abs(x))/env.x_threshold - 0.5)
    r2 = ((env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5) * 1.5
    return r1 + r2
