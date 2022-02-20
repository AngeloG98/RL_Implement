import gym
import random

class PolicyIter():
    def __init__(self, env, gamma = 0.8, error = 0.00001, max_iter = 1000) -> None:
        self.env = env
        self.states = env.states
        self.actions = env.actions
        self.terminal_rewards = env.terminal_rewards
        self.states_actions = env.states_actions

        self.v = dict()
        for state in self.states:
            if state not in self.terminal_rewards:
                self.v[state] = 0.0
            else:
                self.v[state] = self.terminal_rewards[state]
        self.pi = dict()
        
        self.gamma = gamma
        self.error = error
        self.max_iter = max_iter

    def policy_init(self):
        for state in self.states:
            if state in self.terminal_rewards:
                self.pi[state] = None
            else:
                self.pi[state] = random.sample(list(self.states_actions[state]), 1)[0]

    def policy_evaluate(self):
        for _ in range(self.max_iter):
            delta = 0.0
            for state in self.states:
                if state not in self.terminal_rewards:
                    self.env.reset()
                    action = self.pi[state]
                    self.env.set_state(state)
                    next_state, reward, is_terminal, info = self.env.step(action)
                    self.env.reset()
                    v_last = self.v[state]
                    self.v[state] = reward + self.gamma*self.v[next_state]
                    delta = max(delta, abs(self.v[state] - v_last))
            if delta <= self.error:
                break

    def policy_improve(self):
        policy_stable = True
        for state in self.states:
            if state not in self.terminal_rewards:
                max_action = list(self.states_actions[state])[0]
                max_q = -999999.0
                for action in list(self.states_actions[state]):
                    self.env.reset()
                    self.env.set_state(state)
                    next_state, reward, is_terminal, info = self.env.step(action)
                    self.env.reset()
                    q_value = reward + self.gamma*self.v[next_state]
                    if q_value > max_q:
                        max_action = action
                        max_q = q_value
                if self.pi[state] != max_action:
                    policy_stable = False
                    self.pi[state] = max_action
        return policy_stable

    def forward(self, iter):
        self.env.reset()
        for _ in range(int(self.max_iter/50)):
            env.render()
            if _ == 0:
                print("iteration:{}, init_state: {}".format(iter, self.env.state))
            next_state, reward, is_terminal, info = self.env.step(self.pi[self.env.state])
            print("iteration:{}, next_state:{}, reward:{}, is_terminal:{}".format(iter, next_state, reward, is_terminal))
    
    def policy_iterate(self):
        self.policy_init()
        count = 0
        for _ in range(self.max_iter):
            # self.forward(_)
            self.policy_evaluate()
            policy_stable = self.policy_improve()
            if policy_stable == True:
                count += 1
            if count >= 10:
                print("Policy now stabled.")
                break


class ValueIter():
    def __init__(self, env, gamma = 0.8, error = 0.00001, max_iter = 1000) -> None:
        self.env = env
        self.states = env.states
        self.actions = env.actions
        self.terminal_rewards = env.terminal_rewards
        self.states_actions = env.states_actions

        self.v = dict()
        for state in self.states:
            if state not in self.terminal_rewards:
                self.v[state] = 0.0
            else:
                self.v[state] = self.terminal_rewards[state]
        self.pi = dict()
        
        self.gamma = gamma
        self.error = error
        self.max_iter = max_iter
    
    def policy_init(self):
        for state in self.states:
            if state in self.terminal_rewards:
                self.pi[state] = None
            else:
                self.pi[state] = random.sample(list(self.states_actions[state]), 1)[0]

    def policy_evaluate(self): # 1 
        delta = 0.0
        values_stable = True
        for state in self.states:
            if state not in self.terminal_rewards:
                self.env.reset()
                action = self.pi[state]
                self.env.set_state(state)
                next_state, reward, is_terminal, info = self.env.step(action)
                self.env.reset()
                v_last = self.v[state]
                self.v[state] = reward + self.gamma*self.v[next_state]
                delta = max(delta, abs(self.v[state] - v_last))
        if delta >= self.error:
            values_stable = False
        return values_stable

    def policy_improve(self):
        for state in self.states:
            if state not in self.terminal_rewards:
                max_action = list(self.states_actions[state])[0]
                max_q = -999999.0
                for action in list(self.states_actions[state]):
                    self.env.reset()
                    self.env.set_state(state)
                    next_state, reward, is_terminal, info = self.env.step(action)
                    self.env.reset()
                    q_value = reward + self.gamma*self.v[next_state]
                    if q_value > max_q:
                        max_action = action
                        max_q = q_value
                self.pi[state] = max_action

    def forward(self, iter):
        self.env.reset()
        for _ in range(int(self.max_iter/50)):
            env.render()
            if _ == 0:
                print("iteration:{}, init_state: {}".format(iter, self.env.state))
            next_state, reward, is_terminal, info = self.env.step(self.pi[self.env.state])
            print("iteration:{}, next_state:{}, reward:{}, is_terminal:{}".format(iter, next_state, reward, is_terminal))
    
    def policy_iterate(self):
        self.policy_init()
        count = 0
        for _ in range(self.max_iter):
            # self.forward(_)
            values_stable = self.policy_evaluate()
            if values_stable == True:
                count += 1
            if count >= 10:
                print("Value now stabled.")
                break
            self.policy_improve()
    
        
if __name__ == "__main__":
    env = gym.make('GridWorld-v0')
    # PI = PolicyIter(env)
    # PI.policy_iterate()
    # for _ in range(10):
    #     PI.forward("Test")
    VI = ValueIter(env)
    VI.policy_iterate()
    for _ in range(10):
        VI.forward("Test")
    print()
    env.close()
