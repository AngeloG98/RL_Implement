import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from dqn import DQN_agent
from double_dqn import double_DQN_agent
from dueling_dqn import dueling_DQN_agent
from utils import *

env = gym.make('Pendulum-v1')
input_dim = env.observation_space.shape[0]
output_dim = 11

seed = 1423
layer_sizes = [input_dim, 64, output_dim]
lr = 1e-3
gamma = 0.95
sync_freq = 5
exp_replay_size = 256
agent = DQN_agent(seed, layer_sizes, lr, gamma, sync_freq, exp_replay_size)
# agent = double_DQN_agent(seed, layer_sizes, lr, gamma, sync_freq, exp_replay_size)
# agent = dueling_DQN_agent(seed, layer_sizes, lr, gamma, sync_freq, exp_replay_size)

episodes = 8000
epsilon = 1.0
batch_size = 16
loss_list, reward_list, step_list, epsilon_list = [], [], [], []

EPSILON_DEC = 1/3000
EPSILON_MIN = 1/200

LEARN_FREQ = exp_replay_size/2
LEARN_TIMES = 4
collect_count = exp_replay_size/2

SAVE_COUNT_THR = 5
SAVE_REWARD_THR = -0.03
save_count = 0

for episode in tqdm(range(episodes+1)):
    loss_sum, step, step_reward_list, is_terminal = 0, 0, [], False
    state = env.reset()
    while not is_terminal:
        # env.render()
        action, _ = agent.get_action(state, output_dim, epsilon)
        state_, reward, is_terminal, info = env.step(action_undiscre(env, output_dim, action))

        reward = reward/10 

        agent.store_memory(state, action, reward, state_, is_terminal)
        state = state_
    
        if len(agent.exp_replay_mem) >= exp_replay_size:
            if collect_count > LEARN_FREQ:
                collect_count = 0
                for _ in range(LEARN_TIMES):
                    loss = agent.learn(batch_size=16)
                    loss_sum += loss
        step_reward_list.append(reward)
        step += 1
        collect_count += 1
    
    if sum(step_reward_list[-10:]) >= SAVE_REWARD_THR:
        save_count += 1
        if save_count >= SAVE_COUNT_THR:
            print("Saving trained model...")
            agent.save_trained_model("2_Deep_ValueBased/Model/"+ agent.name +"_DQN-Pendulum-v1" + "_episode_" + str(episode)+".pth")
            break
    else:
        save_count = 0
        
    if epsilon > EPSILON_MIN:
        epsilon -= EPSILON_DEC

    if step != 0:
        loss_list.append(loss_sum / step)
        reward_list.append(sum(step_reward_list))
        step_list.append(step)
        epsilon_list.append(epsilon)
env.close()

if save_count < SAVE_COUNT_THR:
    print("Saving trained model...")
    agent.save_trained_model("2_Deep_ValueBased/Model/"+ agent.name +"_DQN-Pendulum-v1" + "_episode_" + str(episodes)+".pth")

plt.subplot(221)
plt.plot(loss_list)
plt.subplot(222)
plt.plot(step_list)
plt.subplot(223)
plt.plot(reward_list)
plt.subplot(224)
plt.plot(epsilon_list)
plt.savefig("2_Deep_ValueBased/Fig/"+ agent.name +"_DQN-Pendulum-v1_plot.jpg")
plt.show()
