from os import environ
import gym
import random
import numpy as np

class MonteCarloPI():
    def __init__(self, env, episodes = 10000, max_step = 100, max_iter = 20, gamma = 0.99, epsilon = 0.4) -> None:
        self.env = env
        self.episodes = episodes
        self.max_step = max_step
        self.max_iter = max_iter
        self.gamma = gamma
        self.epsilon = epsilon
        self.pi = dict()
    
    def policy_init(self):
        for state in self.env.states:
            if state in self.env.terminal_rewards:
                self.pi[state] = None
            else:
                self.pi[state] = random.sample(self.env.actions, 1)[0]

    def mc_policy_evaluate(self):
        self.Q_s_a = np.zeros((len(self.env.states), len(self.env.actions)))
        self.N_s_a = np.zeros((len(self.env.states), len(self.env.actions)))
        for i in range(self.episodes):
            s_sample = []
            a_sample = []
            r_sample = []
            self.env.reset()
            state = self.env.state
            
            # sample an episode
            for t in range(self.max_step):
                s_sample.append(state)
                
                if random.uniform(0, 1) > self.epsilon:
                    action = self.pi[state]
                else:
                    action = random.choice(self.env.actions)
                a_sample.append(action)
                
                state, reward, is_terminal, info = self.env.step(action)
                r_sample.append(reward)

                if is_terminal:
                    break
            
            # incremental monte carlo
            for t in range(len(s_sample)):
                s = s_sample[t]
                a = a_sample[t]
                r = r_sample[t]
                idx_s, idx_a = s-1, self.env.actions.index(a)
                
                self.N_s_a[idx_s][idx_a] += 1
                G_i_t = 0.0
                for k in range(t, len(s_sample)):
                    G_i_t += self.gamma**(k - t) * r_sample[k]
                self.Q_s_a[idx_s][idx_a] += (1/self.N_s_a[idx_s][idx_a]) * (G_i_t - self.Q_s_a[idx_s][idx_a])
        
    def mc_policy_improve(self):
        policy_stable = True
        for state in self.env.states:
            if state not in self.env.terminal_rewards:
                pi_last = self.pi[state]
                self.pi[state] = self.env.actions[np.argmax(self.Q_s_a[state-1])]
                if pi_last != self.pi[state]:
                    policy_stable = False
        return policy_stable

    def forward(self, state, iter):
        self.env.reset()
        self.env.set_state(state)
        for _ in range(int(self.max_step/5)):
            env.render()
            if _ == 0:
                print("iteration:{}, init_state: {}".format(iter, self.env.state))
            next_state, reward, is_terminal, info = self.env.step( self.env.actions[np.argmax(self.Q_s_a[self.env.state-1])] )
            # print("iteration:{}, next_state:{}, reward:{}, is_terminal:{}".format(iter, next_state, reward, is_terminal))
        return reward, is_terminal

    def mc_policy_iterate(self):
        self.policy_init()
        count = 0
        for _ in range(self.max_iter):
            self.mc_policy_evaluate()
            policy_stable = self.mc_policy_improve()
            if policy_stable == True:
                count += 1
            if count >= 2:
                print("Policy now stabled. (iteration: {})".format((_)))
                break


if __name__ == "__main__":
    env = gym.make('GridWorld-v0')
    mc = MonteCarloPI(env)
    mc.mc_policy_iterate()
    for state in env.states:
        if state not in env.terminal_rewards:
            reward, is_terminal = mc.forward(state, "Test")
            if reward > 0 and is_terminal:
                print("state {} sucess :), reward is {}".format(state, reward))
            else:
                print("state {} fail :(, reward is {}".format(state, reward))
    print()
    env.close()