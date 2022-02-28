import random
from collections import deque, namedtuple

# class ReplayMemory():
#     def __init__(self, capacity) -> None:
#         self.capacity = capacity
#         self.memory = deque(maxlen=capacity)
#         self.Transition = namedtuple('Transition',(('state', 'action', 'reward', 'state_', 'is_terminal')))
    
#     def __len__(self):
#         return len(self.memory)

#     def push_pop(self, *args):
#         self.memory.append(self.Transition(*args))
    
#     def unzip(self, transitions):
#         return self.Transition(*zip(*transitions))

#     def sample(self, batch_size):
#         data = random.sample(self.memory, batch_size)
#         return self.unzip(data)

#     def trajectory(self, batch_size):
#         data = self.memory[-batch_size:]
#         return self.unzip(data)


class ReplayMemory():
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        # self.Transition = namedtuple('Transition',(('state', 'action', 'reward', 'state_', 'is_terminal')))
    
    def __len__(self):
        return len(self.memory)

    def push_pop(self, data):
        self.memory.append(data)

    def sample(self, batch_size):
        data = random.sample(self.memory, batch_size)
        return data

    def trajectory(self, batch_size):
        data = self.memory[-batch_size:]
        return data

class Memory_Buffer(object):
    def __init__(self, memory_size=100000):
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0
        
    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size: # buffer not full
            self.buffer.append(data)
        else: # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            
            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done= data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones
    
    def size(self):
        return len(self.buffer)


