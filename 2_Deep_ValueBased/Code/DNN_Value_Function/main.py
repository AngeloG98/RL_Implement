import gym
import matplotlib.pyplot as plt

from tqdm import tqdm
from dqn import DQN_agent

seed = 1423
env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

layer_sizes = [input_dim, 64, output_dim]
lr = 1e-3
gamma = 0.95
sync_freq = 5
exp_replay_size = 256
agent = DQN_agent(seed, layer_sizes, lr, gamma, sync_freq, exp_replay_size)

episodes = 10000
epsilon = 1.0
batch_size = 16
loss_list, reward_list, step_list, epsilon_list = [], [], [], []

index = 128
save_count = 0
for i in tqdm(range(episodes)):
    loss_sum, step, reward_sum, is_terminal = 0, 0, 0, False
    state = env.reset()
    while not is_terminal:
        action = agent.get_action(state, env.action_space.n, epsilon)
        state_, reward, is_terminal, info = env.step(action)
        x, x_dot, theta, theta_dot = state_
        r1 = ((env.x_threshold - abs(x))/env.x_threshold - 0.5)
        r2 = ((env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5) * 1.5
        reward = r1 + r2

        agent.store_memory(state, action, reward, state_, is_terminal)
        # agent.store_memory([state, action, reward, state_, is_terminal])
        state = state_
        index += 1

        if len(agent.exp_replay_mem) >= exp_replay_size:
            # loss = agent.learn(batch_size)
            # loss_sum += loss
            if index > 128:
                index = 0
                for j in range(4):
                    loss = agent.learn(batch_size=16)
                    loss_sum += loss
        reward_sum += reward
        step += 1
    
    if step >= 8000:
        save_count += 1
        if save_count >= 3:
            agent.save_trained_model("2_Deep_ValueBased/Model/cartpole-dqn-my.pth")
            break
    else:
        save_count = 0
        
    if epsilon > 0.05:
        epsilon -= (1 / 5000)

    if step != 0:
        loss_list.append(loss_sum / step)
        reward_list.append(reward_sum)
        step_list.append(step)
        epsilon_list.append(epsilon)

print("Saving trained model")
agent.save_trained_model("2_Deep_ValueBased/Model/cartpole-dqn-my.pth")
plt.subplot(221)
plt.plot(loss_list)
plt.subplot(222)
plt.plot(step_list)
plt.subplot(223)
plt.plot(reward_list)
plt.subplot(224)
plt.plot(epsilon_list)
plt.show()
