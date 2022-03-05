import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pg import PG_agent
from utils import reward_func

env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

configs = {
    'agent': 'PG',
    'layer_sizes': [input_dim, 64, output_dim],
    'lr': 1e-3,
    'gamma': 0.99,
    'Type': ['trajectory', 'future'], # trajectory/step  total/future
    'max_episode': int(1e6),
    'print_freq': 20,
    'save_freq': 200
}

agent = PG_agent(
    configs['layer_sizes'],
    configs['gamma'],
    configs['lr']
)

loss_list, reward_list, step_list= [], [], []
for episode in range(configs['max_episode']):
    reward_sum, step, is_terminal = 0, 0, False
    state = env.reset()

    while not is_terminal:
        # env.render()
        action = agent.get_action(state)
        state_, reward, is_terminal, info = env.step(action)
        # reward = reward_func(env, state_)
        agent.store_traj(state, action, reward)
        
        state = state_
        reward_sum += reward
        step += 1

    loss = agent.learn(configs['Type'])
    agent.reset_traj()
    
    loss_list.append(loss)
    reward_list.append(reward_sum)
    step_list.append(step)

    if episode % configs['print_freq'] == 0:
        if len(reward_list) > 1:
            smooth_reward = 0.1*reward_list[-1] + 0.9*np.mean(reward_list[-100:-1])
        else:
            smooth_reward = reward_list[-1]
        print("====================================================")
        print("train model: " + agent.name)
        print("episode: {}".format(episode))
        print("episode step: {}".format(step))
        print("episode loss: {}".format(round(loss, 4)))
        print("smooth episode reward: {}".format(round(smooth_reward, 2)))

    if episode % configs['save_freq'] == 0 and episode != 0:
        # model
        model_filename = "3_PolicyBased/Model/"+agent.name+"_"+env.env.spec.id+"_episode_"+str(episode)+".pth"
        agent.save_trained_model(model_filename)
        # fig
        fig_filename = "3_PolicyBased/Fig/"+agent.name+"_"+env.env.spec.id+"_episode_"+str(episode)+"_plot.jpg"
        fig = plt.figure(figsize=(9,6))
        fig.suptitle(agent.name+"_"+env.env.spec.id+"_episode_"+str(episode))
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 1, 2)
        ax1.plot(loss_list)
        ax2.plot(step_list)
        ax3.plot(reward_list)
        ax1.set_title("episode loss")
        ax2.set_title("episode step")
        ax3.set_title("episode reward")
        plt.tight_layout()
        plt.savefig(fig_filename, dpi=500)
