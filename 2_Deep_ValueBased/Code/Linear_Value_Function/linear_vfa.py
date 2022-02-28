import abc
import gym
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sutton_tile import *

class LinearVFA(metaclass= abc.ABCMeta):
    def __init__(self, env, num_tiles = 8, param_vector_size = 32768, episodes = 1000, max_step = 1000, lr = 0.1, gamma = 0.9, epsilon = 0.1) -> None:
        self.env = env
        self.episodes = episodes
        self.max_step = max_step
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.weights = np.zeros((param_vector_size,))
        self.action_space = [i for i in range(env.action_space.n)]
        self.obs_space = list(zip(env.observation_space.high, env.observation_space.low))

        self.num_tiles = num_tiles
        self.iht = IHT(param_vector_size)
        self.param_vector_size = param_vector_size

    def get_feature_vector(self, state, action):
        state_tiles = []
        for i in range(len(state)):
            if self.obs_space[i][0] > 10e5: # if obs_space is inf, manully adjust the range
                # adjustment for "CartPole-v0"
                if str(self.env.env.spec.id) == "CartPole-v0":
                    if i == 1: 
                        state_tiles.append(self.num_tiles * state[i] / (1 + 1)) # 'hyperparameter' here
                    if i == 3:
                        state_tiles.append(self.num_tiles * state[i] / (3 + 3)) # 'hyperparameter' here
            else:
                state_tiles.append(self.num_tiles * state[i] / (self.obs_space[i][0]-self.obs_space[i][1]))

        indices = tiles(self.iht, self.num_tiles, state_tiles, action)
        feature_vector = np.zeros((self.param_vector_size,))
        feature_vector[indices] = 1 # one-hot 
        return feature_vector

    def epsilon_greed(self, state):
        if random.uniform(0, 1.0) > self.epsilon:
            values = []
            for a in self.action_space:
                feature_vector = self.get_feature_vector(state, [a])
                values.append(self.compute_value(feature_vector))
            action = [values.index(max(values))]
        else:
            action = [random.choice(self.action_space)]
        return action

    def greed(self, state):
        values = []
        for a in self.action_space:
            feature_vector = self.get_feature_vector(state, [a])
            values.append(self.compute_value(feature_vector))
        action = [values.index(max(values))]
        return action

    def compute_value(self, feature_vector):
        return np.dot(self.weights, feature_vector)

    def learn(self, s, a, r, s_, a_, is_terminal):
        feature_vector = self.get_feature_vector(s, a)
        feature_vector_ = self.get_feature_vector(s_, a_)
        if is_terminal:
            q_target = r
        else:
            q_target = r + self.gamma * self.compute_value(feature_vector_)
        self.weights += self.lr/self.num_tiles * (q_target - self.compute_value(feature_vector)) * feature_vector
    
    def forward(self, test_step = 10000, filename = None):
        if filename != None:
            model = np.load(filename+".npz")
            self.weights = model["w"]
            with open(filename+".pkl", "rb") as f:
                self.iht.dictionary = pickle.load(f)
        
        self.env.reset()
        state = self.env.state
        for step in range(test_step):
            self.env.render()
            action = self.greed(state)
            state, reward, is_terminal, info = self.env.step(action[0])
            if is_terminal:
                print("step:{}".format(step))
                break

    @abc.abstractclassmethod
    def train(self):
        pass

class Sarsa_VFA(LinearVFA):
    def __init__(self, env, num_tiles=8, param_vector_size=32768, episodes=1000, max_step=1000, lr=0.1, gamma=0.9, epsilon=0.1) -> None:
        super().__init__(env, num_tiles, param_vector_size, episodes, max_step, lr, gamma, epsilon)

    def train(self):
        episode_reward_list = []
        for episode in range(self.episodes+1):
            self.env.reset()
            state = self.env.state
            action = self.epsilon_greed(state)

            episode_reward = 0
            for step in range(self.max_step):
                # self.env.render()
                state_, reward, is_terminal, info = self.env.step(action[0])
                action_ = self.epsilon_greed(state_)
                
                # break for "MountainCar-v0"
                if state_[0] >= 0.5:
                    is_terminal = True

                # reward function for "CartPole-v0"
                if str(self.env.env.spec.id) == "CartPole-v0":
                    x, x_dot, theta, theta_dot = state_
                    r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.5
                    r2 = ((self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5)*1.2
                    reward = r1 + r2
                
                episode_reward += reward
                self.learn(state, action, reward, state_, action_, is_terminal)
                
                if str(self.env.env.spec.id) == "CartPole-v0":
                    # break for "CartPole-v0"
                    if self.env.x_threshold < abs(x) or self.env.theta_threshold_radians*5 < abs(theta):
                        break
                else:
                    if is_terminal:
                        break

                state = state_
                action  = action_
            
            episode_reward_list.append(episode_reward)
            if episode % 100 == 0 and episode != 0:
                self.save_weight(episode)
            print("episode:{}, reward_sum: {}".format(episode, episode_reward))
        return episode_reward_list

    def save_weight(self, episode):
        np.savez("2_Deep_ValueBased/Model/Sarsa_VFA-" + str(self.env.env.spec.id) + "_episode_" + str(episode)+".npz", w = self.weights)
        with open("2_Deep_ValueBased/Model/Sarsa_VFA-" + str(self.env.env.spec.id) + "_episode_" + str(episode)+".pkl", "wb") as f:
            pickle.dump(self.iht.dictionary, f)

class Qlearning_VFA(LinearVFA):
    def __init__(self, env, num_tiles=8, param_vector_size=32768, episodes=1000, max_step=1000, lr=0.1, gamma=0.9, epsilon=0.1) -> None:
        super().__init__(env, num_tiles, param_vector_size, episodes, max_step, lr, gamma, epsilon)

    def train(self):
        episode_reward_list = []
        for episode in range(self.episodes+1):
            self.env.reset()
            state = self.env.state
            
            episode_reward = 0
            for step in range(self.max_step):
                # self.env.render()
                action = self.epsilon_greed(state)
                state_, reward, is_terminal, info = self.env.step(action[0])
                action_ = self.greed(state_)

                # break for "MountainCar-v0"
                if state_[0] >= 0.5:
                    is_terminal = True

                # reward function for "CartPole-v0"
                if str(self.env.env.spec.id) == "CartPole-v0":
                    x, x_dot, theta, theta_dot = state_
                    r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.5
                    r2 = ((self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5)*1.2
                    reward = r1 + r2
                
                episode_reward += reward
                self.learn(state, action, reward, state_, action_, is_terminal)
                
                if str(self.env.env.spec.id) == "CartPole-v0":
                    # break for "CartPole-v0"
                    if self.env.x_threshold < abs(x) or self.env.theta_threshold_radians*5 < abs(theta):
                        break
                else:
                    if is_terminal:
                        break

                state = state_
            
            episode_reward_list.append(episode_reward)
            if episode % 100 == 0 and episode != 0:
                self.save_weight(episode)
            print("episode:{}, reward_sum: {}".format(episode, episode_reward))
        return episode_reward_list

    def save_weight(self, episode):
        np.savez("2_Deep_ValueBased/Model/Qlearning_VFA-" + str(self.env.env.spec.id) + "_episode_" + str(episode)+".npz", w = self.weights)
        with open("2_Deep_ValueBased/Model/Qlearning_VFA-" + str(self.env.env.spec.id) + "_episode_" + str(episode)+".pkl", "wb") as f:
            pickle.dump(self.iht.dictionary, f)

if __name__ == "__main__":
    # parameters
    episodes = 400
    max_step = 1000
    lr = 0.5
    gamma = 0.9
    epsilon = 0.3 # MountainCar-v0 0.0 / CartPole-v0 0.3
    num_tiles = 8 # 8 16
    param_vector_size = 32768 # 32768 1048576
    # environment
    # env = gym.make("MountainCar-v0") # change max_episode_steps=10000
    env = gym.make("CartPole-v0") # change max_episode_steps=10000
    # model
    # lvfa = Sarsa_VFA(env, num_tiles, param_vector_size, episodes, max_step, lr, gamma, epsilon)
    lvfa = Qlearning_VFA(env, num_tiles, param_vector_size, episodes, max_step, lr, gamma, epsilon)
    # train
    episode_reward_list = lvfa.train()
    for _ in range(10):
        lvfa.forward(test_step = 10000)
    plt.plot(episode_reward_list)
    plt.show()
    # test
    lvfa.forward(test_step = 10000, filename = "2_Deep_ValueBased/Model/Sarsa_VFA-CartPole-v0_episode_400")
    # lvfa.forward(test_step = 10000, filename = "2_Deep_ValueBased/Model/Qlearning_VFA-CartPole-v0_episode_400")
    # lvfa.forward(test_step = 10000, filename = "2_Deep_ValueBased/Model-demo/Sarsa_VFA-CartPole-v0_episode_200") # 16 1048576
    # lvfa.forward(test_step = 10000, filename = "2_Deep_ValueBased/Model-demo/Qlearning_VFA-CartPole-v0_episode_800") # 8 32768