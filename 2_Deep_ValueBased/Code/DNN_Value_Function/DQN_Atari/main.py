import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dqn import DQN_agent
from dqn_test import DQN_agent1
from utils import *
from atari_wrappers import make_atari, wrap_deepmind

env = make_atari('PongNoFrameskip-v4')
env = wrap_deepmind(env, scale = False, frame_stack=True)
action_space = env.action_space
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]
state_channel = env.observation_space.shape[2]

seed = 1423
gamma = 0.99

lr = 2e-4
sync_freq = 1000
exp_replay_size = 100000
agent = DQN_agent1(seed, state_channel, action_space, lr, gamma, sync_freq, exp_replay_size)

episodes = 1000000
epsilon = 1.0
batch_size = 32
loss_list, reward_list, step_list, epsilon_list = [], [], [], []

EPSILON_DEC = 1/3000
EPSILON_MIN = 1/200

frame_idx = 0
epsilon_max = 1
epsilon_min = 0.01
eps_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(-1. * frame_idx / eps_decay)
for episode in range(episodes+1):
    loss_sum, step, step_reward_list, is_terminal = 0, 0, [], False
    frame = env.reset()
    while not is_terminal:# env.render()
        epsilon_i = epsilon_by_frame(frame_idx)
        state = agent.observe(frame)
        action, _ = agent.get_action(state, epsilon_i)
        frame_, reward, is_terminal, info = env.step(action)
        # state_ = agent.get_state(frame_)
        
        # agent.store_memory([state[0], action, reward, state_[0], is_terminal])
        agent.memory_buffer.push(frame, action, reward, frame_, is_terminal)
        frame = frame_
    
        # if len(agent.exp_replay_mem) >= exp_replay_size/100:
        if agent.memory_buffer.size() >= exp_replay_size/100:
            loss = agent.learn(batch_size)
            loss_sum += loss
        step_reward_list.append(reward)
        step += 1
        frame_idx += 1

        if frame_idx % (exp_replay_size/100) == 0 and step != 0:
            print("frame_idx: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (frame_idx, np.mean(reward_list[-10:]), loss, epsilon_i, episode))
    
    # if sum(step_reward_list[-10:]) >= SAVE_REWARD_THR:
    #     save_count += 1
    #     if save_count >= SAVE_COUNT_THR:
    #         print("Saving trained model...")
    #         agent.save_trained_model("2_Deep_ValueBased/Model/"+ agent.name +"_DQN-Pendulum-v1" + "_episode_" + str(episode)+".pth")
    #         break
    # else:
    #     save_count = 0
        
    if epsilon > EPSILON_MIN:
        epsilon -= EPSILON_DEC

    if step != 0:
        loss_list.append(loss_sum)
        reward_list.append(sum(step_reward_list))
        step_list.append(step)
        epsilon_list.append(epsilon_i)
env.close()

# if save_count < SAVE_COUNT_THR:
#     print("Saving trained model...")
#     agent.save_trained_model("2_Deep_ValueBased/Model/"+ agent.name +"_DQN-Pendulum-v1" + "_episode_" + str(episodes)+".pth")

# plt.subplot(221)
# plt.plot(loss_list)
# plt.subplot(222)
# plt.plot(step_list)
# plt.subplot(223)
# plt.plot(reward_list)
# plt.subplot(224)
# plt.plot(epsilon_list)
# plt.savefig("2_Deep_ValueBased/Fig/"+ agent.name +"_DQN-Pendulum-v1_plot.jpg")
# plt.show()
