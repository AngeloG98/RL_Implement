import gym
import numpy as np
import matplotlib.pyplot as plt
from atari_wrapper_pytorch import make_atari, wrap_deepmind
from dqn import DQN_agent

configs = {
    "seed": 10,
    "env": "PongNoFrameskip-v4", # use NoFrameskip versions
    "agent": "DQN", # DQN or 

    # agent hyper-parameters
    "gamma": 0.99, # discount factor

    # train hyper-parameters
    "lr": 1e-4,
    "batch_size": 32,
    "max_episode": 500,
    "eps_start": 1.0, # epsilon-greed
    "eps_end": 0.01,
    "eps_decay": 1e5,
    "exp_replay_size": 5000, # experience replay buffer size
    "sync_freq": 1000, # update target network frequence

    # logging parameters
    "video_freq": 50,
    "print_freq": 10,
    "save_freq": 100
}

env = make_atari(configs["seed"], configs["env"])
env = wrap_deepmind(env, frame_stack=True, scale=False)
env = gym.wrappers.Monitor(env, '2_Deep_ValueBased/Video/'+configs["agent"]+'/', video_callable=lambda episode_id: episode_id%configs["video_freq"]==0,force=True)

agent = DQN_agent(
    seed=configs["seed"],
    input_shape=env.observation_space.shape,
    num_actions=env.action_space.n,
    lr=configs["lr"],
    gamma=configs["gamma"],
    sync_freq=configs["sync_freq"],
    exp_replay_size=configs["exp_replay_size"]
)

seed = configs["seed"]
epsilon = configs["eps_start"]
loss_list, reward_list, step_list, epsilon_list = [], [], [], []

for episode in range(configs["max_episode"]):
    loss_sum, reward_sum, step, is_terminal = 0, 0, 0, False
    env.seed(seed) 
    state = env.reset()

    while not is_terminal:
        action, _ = agent.get_action(agent.norm_state(state), epsilon)
        state_, reward, is_terminal, info = env.step(action)

        agent.store_memory(agent.norm_state(state), action, reward, agent.norm_state(state_), is_terminal)
        state = state_

        if len(agent.exp_replay_mem) >= configs["exp_replay_size"]:
            loss = agent.learn(configs["batch_size"])
            loss_sum += loss

        reward_sum += reward
        step += 1
        decay = min(1.0, (sum(step_list)+step) / configs["eps_decay"])
        epsilon = configs["eps_start"] + decay * (configs["eps_end"] - configs["eps_start"])
        epsilon_list.append(epsilon)
    
    seed += 1
    loss_list.append(loss_sum/step)
    reward_list.append(reward_sum)
    step_list.append(step)
    
    if episode % configs["print_freq"] == 0:
        if len(reward_list) > 1:
            smooth_reward = 0.1*reward_list[-1] + 0.9*np.mean(reward_list[-100:-1])
        else:
            smooth_reward = reward_list[-1]
        print("====================================================")
        print("total_steps: {}".format(sum(step_list)))
        print("episode: {}".format(episode))
        print("epsilon: {}".format(round(epsilon_list[-1], 1)))
        print("smooth episode reward: {}".format(round(smooth_reward, 1)))
        print("episode loss:{}".format(loss_list[-1], 1))

    if episode % configs["save_freq"] == 0 and episode != 0:
        # model
        model_filename = "2_Deep_ValueBased/Model/DQN_Atari_pretrained/"+agent.name+"-DQN_"+configs["env"]+"_episode_"+str(episode)+".pth"
        agent.save_trained_model(model_filename)
        # fig
        fig_filename = "2_Deep_ValueBased/Fig/"+agent.name+"-DQN_"+configs["env"]+"_episode_"+str(episode)+"_plot.jpg"
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle(agent.name+"-DQN_"+configs["env"]+"_episode_"+str(episode))
        ax1.plot(loss_list)
        ax1.set_ylabel("loss")
        ax2.plot(step_list)
        ax2.set_ylabel("step")
        ax3.plot(reward_list)
        ax3.set_ylabel("reward")
        ax4.plot(epsilon_list)
        ax4.set_ylabel("epsilon")
        plt.savefig(fig_filename)
