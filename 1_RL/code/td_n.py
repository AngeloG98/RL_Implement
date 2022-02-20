import gym
import random
import numpy as np
from td import TD

class SarsaN(TD):
    def __init__(self, env, n_step = 3, episodes=1000, max_step=100, lr=0.01, gamma=0.9, epsilon=0.1) -> None:
        super().__init__(env, episodes, max_step, lr, gamma, epsilon)
        self.n = n_step

    def compute_G(self, tao, s_sample, a_sample, r_sample, is_terminal):
        G = 0
        for i in range(tao, tao + self.n):
            G += self.gamma**(i-tao) * r_sample[i]
        if not is_terminal:
            idx_s = s_sample[tao + self.n] - 1
            idx_a = self.env.actions.index(a_sample[tao + self.n])
            G += self.gamma**self.n * self.Q_s_a[idx_s][idx_a]
        return G
    
    def nstep_learn(self, s, a, G):
        idx_s, idx_a = s-1, self.env.actions.index(a)
        self.Q_s_a[idx_s][idx_a] += self.lr*(G - self.Q_s_a[idx_s][idx_a])

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
            for t in range(self.max_step):
                state_, reward, is_terminal, info = self.env.step(action)
                r_sample.append(reward)

                action_ = self.epsilon_greed(state_)
                self.learn(state, action, reward, state_, action_, is_terminal)

                state = state_
                action  = action_
                s_sample.append(state)
                a_sample.append(action)

                tao = t - self.n + 1
                if tao >= 0:
                    G = self.compute_G(tao, s_sample, a_sample, r_sample, is_terminal)
                    self.nstep_learn(s_sample[tao], a_sample[tao], G)
                if is_terminal:
                    break

class SarsaN_offpolicy(TD):
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

    def compute_G_rho(self, tao, s_sample, a_sample, r_sample, is_terminal):
        G = 0
        rho = 1
        for i in range(tao, min(tao + self.n, self.episodes)):
            G += self.gamma**(i-tao) * r_sample[i]
            rho *= self.target_policy_probs(s_sample[i], a_sample[i])/self.behaviour_policy_probs(s_sample[i], a_sample[i])
        if not is_terminal:
            idx_s = s_sample[tao + self.n] - 1
            idx_a = self.env.actions.index(a_sample[tao + self.n])
            G += self.gamma**self.n * self.Q_s_a[idx_s][idx_a]
        return G, rho
    
    def nstep_learn(self, s, a, G):
        idx_s, idx_a = s-1, self.env.actions.index(a)
        self.Q_s_a[idx_s][idx_a] += self.lr*(G - self.Q_s_a[idx_s][idx_a])

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
            for t in range(self.max_step):
                state_, reward, is_terminal, info = self.env.step(action)
                r_sample.append(reward)

                action_ = self.epsilon_greed(state_)
                self.learn(state, action, reward, state_, action_, is_terminal)

                state = state_
                action  = action_
                s_sample.append(state)
                a_sample.append(action)

                tao = t - self.n + 1
                if tao >= 0:
                    G, rho = self.compute_G_rho(tao, s_sample, a_sample, r_sample, is_terminal)
                    self.nstep_learn(s_sample[tao], a_sample[tao], G)
                if is_terminal:
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
