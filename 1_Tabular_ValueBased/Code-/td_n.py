import abc
import gym
import random
import numpy as np

class TDn(metaclass= abc.ABCMeta):
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

    def nstep_learn(self, s, a, G):
        idx_s, idx_a = s-1, self.env.actions.index(a)
        self.Q_s_a[idx_s][idx_a] += self.lr*(G - self.Q_s_a[idx_s][idx_a])

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

class SarsaN(TDn):
    def __init__(self, env, n_step = 3, episodes=1000, max_step=100, lr=0.01, gamma=0.9, epsilon=0.1) -> None:
        super().__init__(env, episodes, max_step, lr, gamma, epsilon)
        self.n = n_step

    def compute_G(self, tao, s_sample, a_sample, r_sample, T):
        G = 0
        for i in range(tao, min(tao + self.n, T)):
            G += self.gamma**(i-tao) * r_sample[i]
        if tao + self.n < T:
            idx_s = s_sample[tao + self.n] - 1
            idx_a = self.env.actions.index(a_sample[tao + self.n])
            G += self.gamma**self.n * self.Q_s_a[idx_s][idx_a]
        return G

    def update(self):
        for episode in range(self.episodes):
            s_sample = []
            a_sample = []
            r_sample = []

            self.env.reset()
            state = self.env.state
            action = self.epsilon_greed(state)

            s_sample.append(state)
            a_sample.append(action)

            T = self.max_step
            t = 0
            for t in range(self.max_step):
                if t < T:
                    state_, reward, is_terminal, info = self.env.step(action)
                    r_sample.append(reward)

                    action_ = self.epsilon_greed(state_)

                    state = state_
                    action  = action_
                    s_sample.append(state)
                    a_sample.append(action)

                    if is_terminal:
                        T = t + 1

                tao = t - self.n + 1
                if tao >= 0:
                    G = self.compute_G(tao, s_sample, a_sample, r_sample, T)
                    self.nstep_learn(s_sample[tao], a_sample[tao], G)
                if tao == T - 1:
                    break
                

class SarsaN_offpolicy(TDn):
    def __init__(self, env, n_step = 3, episodes=1000, max_step=100, lr=0.01, gamma=0.9, epsilon=0.1) -> None:
        super().__init__(env, episodes, max_step, lr, gamma, epsilon)
        self.n = n_step

    def target_policy_probs(self, s, a):
        idx_s = s-1
        if a == self.env.actions[np.argmax(self.Q_s_a[idx_s])]:
            return 1
        else:
            return 0

    def behaviour_policy_probs(self, s, a):
        idx_s = s-1
        if a == self.env.actions[np.argmax(self.Q_s_a[idx_s])]:
            return (1 - self.epsilon) + 0.25*self.epsilon
        else:
            return 0.25*self.epsilon

    def compute_G_rho(self, tao, s_sample, a_sample, r_sample, T):
        G = 0
        rho = 1
        for i in range(tao, min(tao + self.n, T)):
            G += self.gamma**(i-tao) * r_sample[i]
            rho *= self.target_policy_probs(s_sample[i], a_sample[i])/self.behaviour_policy_probs(s_sample[i], a_sample[i])
        if tao + self.n < T:
            idx_s = s_sample[tao + self.n] - 1
            idx_a = self.env.actions.index(a_sample[tao + self.n])
            G += self.gamma**self.n * self.Q_s_a[idx_s][idx_a]
        return G, rho

    def update(self):
        for episode in range(self.episodes):
            s_sample = []
            a_sample = []
            r_sample = []

            self.env.reset()
            state = self.env.state
            action = self.epsilon_greed(state)

            s_sample.append(state)
            a_sample.append(action)

            T = self.max_step
            t = 0
            for t in range(self.max_step):
                if t < T:
                    state_, reward, is_terminal, info = self.env.step(action)
                    r_sample.append(reward)

                    action_ = self.epsilon_greed(state_)

                    state = state_
                    action  = action_
                    s_sample.append(state)
                    a_sample.append(action)

                    if is_terminal:
                        T = t + 1

                tao = t - self.n + 1
                if tao >= 0:
                    G, rho = self.compute_G_rho(tao, s_sample, a_sample, r_sample, T)
                    self.nstep_learn(s_sample[tao], a_sample[tao], G)
                if tao == T - 1:
                    break

if __name__ == "__main__":
    env = gym.make('GridWorld-v0')
    # td = SarsaN(env)
    td = SarsaN_offpolicy(env)
    td.update()
    for state in env.states:
        if state not in env.terminal_rewards:
            reward, is_terminal = td.forward(state, "Test")
            if reward > 0 and is_terminal:
                print("state {} sucess :), reward is {}".format(state, reward))
            else:
                print("state {} fail :(, reward is {}".format(state, reward))
    env.close()
