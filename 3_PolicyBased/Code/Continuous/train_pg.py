import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pg_baseline import PG_B_agent

env = gym.make('Pendulum-v1') # max_episode_steps=1000 Important!
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]

configs = {
    'agent': 'PG_baseline',
    'policy_layer_size': [input_dim, 64, output_dim],
    'value_layer_size': [input_dim, 32, 1],
    'p_lr': 8e-4,
    'v_lr': 1e-3,
    'gamma': 0.99,
    'max_episode': int(1e6),
    'print_freq': 20,
    'save_freq': 200
}

agent = PG_B_agent(
    configs['policy_layer_size'],
    configs['value_layer_size'],
    configs['gamma'],
    configs['p_lr'],
    configs['v_lr']
)

loss_list, reward_list, step_list= [], [], []
for episode in range(configs['max_episode']+1):
    reward_sum, step, is_terminal = 0, 0, False
    state = env.reset()

    while not is_terminal:
        # env.render()
        action = agent.get_action(state)
        state_, reward, is_terminal, info = env.step([action])
        reward = reward/10
        agent.store_traj(state, action, reward)
        
        state = state_
        reward_sum += reward
        step += 1

    loss = agent.learn()
    agent.reset_traj()
    
    loss_list.append(loss)
    reward_list.append(reward_sum)
    step_list.append(step)

    if episode % configs['print_freq'] == 0 and episode != 0:
        if len(reward_list) > 1:
            smooth_reward = 0.1*reward_list[-1] + 0.9*np.mean(reward_list[-100:-1])
        else:
            smooth_reward = reward_list[-1]
        print("====================================================")
        print("train model: " + agent.name)
        print("episode: {}".format(episode))
        print("episode step: {}".format(step))
        if configs['agent'] == 'PG':
            print("episode loss: {}".format(round(loss, 4)))
        else:
            print("episode value loss: {}".format(round(loss[1], 4)))
            print("episode policy loss: {}".format(round(loss[0], 4)))
        print("smooth episode reward: {}".format(round(smooth_reward, 2)))

    if episode % configs['save_freq'] == 0 and episode != 0:
        # model
        model_filename = "3_PolicyBased/Model/"+agent.name+"_"+env.env.spec.id+"_episode_"+str(episode)+".pth"
        agent.save_trained_model(model_filename)
        # fig
        fig_filename = "3_PolicyBased/Fig/"+agent.name+"_"+env.env.spec.id+"_episode_"+str(episode)+"_plot.jpg"
        fig = plt.figure(figsize=(9,6))
        fig.suptitle(agent.name+"_"+env.env.spec.id+"_episode_"+str(episode))
        if configs['agent'] == 'PG':
            ax1 = plt.subplot(2, 2, 1)
            ax2 = plt.subplot(2, 2, 2)
            ax3 = plt.subplot(2, 1, 2)
            ax1.plot(loss_list)
            ax2.plot(step_list)
            ax3.plot(reward_list)
            ax1.set_title("episode loss")
            ax2.set_title("episode step")
            ax3.set_title("episode reward")
        else:
            ax1 = plt.subplot(2, 2, 1)
            ax2 = plt.subplot(2, 2, 2)
            ax3 = plt.subplot(2, 2, 3)
            ax4 = plt.subplot(2, 2, 4)
            ax1.plot(np.array(loss_list)[:,0])
            ax2.plot(np.array(loss_list)[:,1])
            ax3.plot(step_list)
            ax4.plot(reward_list)
            ax1.set_title("episode policy loss")
            ax2.set_title("episode value loss")
            ax3.set_title("episode step")
            ax4.set_title("episode reward")
        plt.tight_layout()
        plt.savefig(fig_filename, dpi=500)
