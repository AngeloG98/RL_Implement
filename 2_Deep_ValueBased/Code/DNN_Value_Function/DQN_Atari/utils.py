import random
import numpy as np
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


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class Prior_ReplayMemory(object):
    def __init__(self, memory_size=1000, a = 0.6, e = 0.01):
        self.tree =  SumTree(memory_size)
        self.memory_size = memory_size
        self.prio_max = 0.1
        self.a = a
        self.e = e
        
    def push(self, state, action, reward, state_, is_terminal):
        data = (state, action, reward, state_, is_terminal)
        p = (np.abs(self.prio_max) + self.e) ** self.a
        self.tree.add(p, data)

    def sample(self, batch_size):
        states, actions, rewards, states_, is_terminals = [], [], [], [], []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            state, action, reward, state_, is_terminal = data
            states.append(state[np.newaxis,:])
            actions.append(action)
            rewards.append(reward)
            states_.append(state_[np.newaxis,:])
            is_terminals.append(is_terminal)
            priorities.append(p)
            idxs.append(idx)
        return idxs, np.concatenate(states), actions, rewards, np.concatenate(states_), is_terminals
    
    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p) 
        
    def size(self):
        return self.tree.n_entries
