import gym
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from dqn import DQN_agent
from utils import *

env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

seed = 1423
layer_sizes = [input_dim, 64, output_dim]
lr = 1e-3
gamma = 0.95
sync_freq = 5
exp_replay_size = 256
agent = DQN_agent(seed, layer_sizes, lr, gamma, sync_freq, exp_replay_size)
agent.load_pretrained_model("2_Deep_ValueBased/Model/DQN-CartPole-v0_episode_4151.pth")

TEST_TIMES = 3
TEST_MAX_STEP = 1000

reward_list, step_list = [], []
for i in tqdm(range(TEST_TIMES)):
    step, reward_sum, is_terminal = 0, 0, False
    state = env.reset()
    while not is_terminal:
    # for _ in range(TEST_MAX_STEP):
    #     time.sleep(0.01)
    #     env.render()
        action = agent.get_action(state, env.action_space.n, epsilon = 0.0)
        state_, reward, is_terminal, info = env.step(action)
        reward = reward_func(env, state_)

        state = state_
        reward_sum += reward
        step += 1

    reward_list.append(reward_sum)
    step_list.append(step)
env.close()

plt.subplot(121)
plt.plot(step_list)
plt.subplot(122)
plt.plot(reward_list)
plt.show()