import abc
import gym
import random
import numpy as np

class TD(metaclass= abc.ABCMeta):
    def __init__(self, env, episodes = 10000, max_step = 100, lr = 0.01, gamma = 0.9, epsilon = 0.1) -> None:
        self.env = env
        self.episodes = episodes
        self.max_step = max_step
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q_s_a = np.zeros((len(self.env.states), len(self.env.actions)))

    def epsilon_greed(self, state):
        idx_s = state - 1 #
        if random.uniform(0, 1) > self.epsilon:
            action = self.env.actions[np.argmax(self.Q_s_a[idx_s])]
        else:
            action = random.choice(self.env.actions)
        return action

    def greed(self, state):
        idx_s = state - 1 #
        return self.env.actions[np.argmax(self.Q_s_a[idx_s])]

    def learn(self, s, a, r, s_, a_, is_terminal):
        idx_s, idx_s_ = s - 1, s_ - 1
        idx_a, idx_a_ = self.env.actions.index(a), self.env.actions.index(a_)
        if is_terminal:
            q_target = r
        else:
            q_target = r + self.gamma*self.Q_s_a[idx_s_][idx_a_]
        self.Q_s_a[idx_s][idx_a] += self.lr*(q_target - self.Q_s_a[idx_s][idx_a])

    @abc.abstractclassmethod
    def update(self):
        pass

    def forward(self, state, iter):
        self.env.reset()
        self.env.set_state(state)
        for _ in range(int(self.max_step/5)):
            self.env.render()
            if _ == 0:
                print("iteration:{}, init_state: {}".format(iter, self.env.state))
            next_state, reward, is_terminal, info = self.env.step( self.env.actions[np.argmax(self.Q_s_a[self.env.state-1])] )
            # print("iteration:{}, next_state:{}, reward:{}, is_terminal:{}".format(iter, next_state, reward, is_terminal))
        return reward, is_terminal


class Sarsa(TD):
    def __init__(self, env, episodes=10000, max_step=100, lr=0.01, gamma=0.9, epsilon=0.1) -> None:
        super().__init__(env, episodes, max_step, lr, gamma, epsilon)

    def update(self):
        for episode in range(self.episodes):
            self.env.reset()
            state = self.env.state
            action = self.epsilon_greed(state)
            for step in range(self.max_step):
                state_, reward, is_terminal, info = self.env.step(action)
                action_ = self.epsilon_greed(state_)
                self.learn(state, action, reward, state_, action_, is_terminal)
                state = state_
                action  = action_


class Qlearning(TD):
    def __init__(self, env, episodes=10000, max_step=100, lr=0.01, gamma=0.9, epsilon=0.1) -> None:
        super().__init__(env, episodes, max_step, lr, gamma, epsilon)
    
    def update(self):
        for episode in range(self.episodes):
            self.env.reset()
            state = self.env.state
            for step in range(self.max_step):
                action = self.epsilon_greed(state)
                state_, reward, is_terminal, info = self.env.step(action)
                action_ = self.greed(state_)
                self.learn(state, action, reward, state_, action_, is_terminal)
                state = state_

class Ex_Sarsa(TD):
    def __init__(self, env, episodes=10000, max_step=100, lr=0.01, gamma=0.9, epsilon=0.1) -> None:
        super().__init__(env, episodes, max_step, lr, gamma, epsilon)

    def Ex_Q(self, idx_s):
        Q_s_expectation = ((1 - self.epsilon) + 0.25*self.epsilon) * max(self.Q_s_a[idx_s])
        for i in range(len(self.env.actions)):
            if i != np.argmax(self.Q_s_a[idx_s]):
                Q_s_expectation += 0.25*self.epsilon*self.Q_s_a[idx_s][i]
        return Q_s_expectation

    def Ex_learn(self, s, a, r, s_, is_terminal):
        idx_s, idx_s_ = s - 1, s_ - 1
        idx_a = self.env.actions.index(a)
        if is_terminal:
            q_target = r
        else:
            q_target = r + self.gamma*self.Ex_Q(idx_s_)
        self.Q_s_a[idx_s][idx_a] += self.lr*(q_target - self.Q_s_a[idx_s][idx_a])
    
    def update(self):
        for episode in range(self.episodes):
            self.env.reset()
            state = self.env.state
            for step in range(self.max_step):
                action = self.epsilon_greed(state)
                state_, reward, is_terminal, info = self.env.step(action)
                self.Ex_learn(state, action, reward, state_, is_terminal)
                state = state_

if __name__ == "__main__":
    env = gym.make('GridWorld-v0')
    # td = Sarsa(env)
    # td = Qlearning(env)
    td = Ex_Sarsa(env)
    td.update()
    for state in env.states:
        if state not in env.terminal_rewards:
            reward, is_terminal = td.forward(state, "Test")
            if reward > 0 and is_terminal:
                print("state {} sucess :), reward is {}".format(state, reward))
            else:
                print("state {} fail :(, reward is {}".format(state, reward))
    env.close()