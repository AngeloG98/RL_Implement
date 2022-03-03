import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from atari_wrapper_pytorch import make_atari, wrap_deepmind
from dqn import DQN_agent
from d3qn_per import D3QN_PER_agent

configs = {
    "seed": 15,
    # use NoFrameskip versions e.g. BreakoutNoFrameskip-v4, PongNoFrameskip-v4
    "env": "PongNoFrameskip-v4", 
    "agent": "D3QN_PER", # DQN or D3QN_PER

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
env = gym.wrappers.Monitor(
    env,
    '2_Deep_ValueBased/Video/'+configs["agent"]+'/',
    video_callable=lambda episode_id: episode_id % configs["video_freq"] == 0,
    force=True
)

if configs["agent"] == "DQN":
    agent = DQN_agent(
        seed=configs["seed"],
        input_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        lr=configs["lr"],
        gamma=configs["gamma"],
        sync_freq=configs["sync_freq"],
        exp_replay_size=configs["exp_replay_size"]
    )
else:
    agent = D3QN_PER_agent(
        seed=configs["seed"],
        input_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        lr=configs["lr"],
        gamma=configs["gamma"],
        sync_freq=configs["sync_freq"],
        exp_replay_size=configs["exp_replay_size"]
    )

epsilon = configs["eps_start"]
loss_list, reward_list, step_list, epsilon_list = [], [], [], []

for episode in range(configs["max_episode"]+1):
    loss_sum, reward_sum, step, is_terminal = 0, 0, 0, False
    state = env.reset()

    while not is_terminal:
        action, _ = agent.get_action(agent.norm_state(state), epsilon)
        state_, reward, is_terminal, info = env.step(action)

        agent.store_memory(agent.norm_state(state), action, reward, agent.norm_state(state_), is_terminal)
        state = state_

        if agent.exp_replay_mem.size() >= configs["exp_replay_size"]:
            loss = agent.learn(configs["batch_size"])
            loss_sum += loss

        reward_sum += reward
        step += 1
        decay = min(1.0, (sum(step_list)+step) / configs["eps_decay"])
        epsilon = configs["eps_start"] + decay * (configs["eps_end"] - configs["eps_start"])
        epsilon_list.append(epsilon)
    
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
        print("epsilon: {}".format(round(epsilon_list[-1], 3)))
        print("episode loss: {}".format(round(loss_list[-1], 6)))
        print("smooth episode reward: {}".format(round(smooth_reward, 2)))

    if episode % configs["save_freq"] == 0 and episode != 0:
        # model
        model_filename = "2_Deep_ValueBased/Model/DQN_Atari_pretrained/"+agent.name+"-DQN_"+configs["env"]+"_episode_"+str(episode)+".pth"
        agent.save_trained_model(model_filename)
        # fig
        fig_filename = "2_Deep_ValueBased/Fig/"+agent.name+"-DQN_"+configs["env"]+"_episode_"+str(episode)+"_plot.jpg"
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9,6))
        fig.suptitle(agent.name+"-DQN_"+configs["env"]+"_episode_"+str(episode))
        ax1.plot(loss_list)
        ax2.plot(step_list)
        ax3.plot(reward_list)
        ax4.plot(epsilon_list)
        ax1.set_title("episode loss")
        ax2.set_title("episode step")
        ax3.set_title("episode reward")
        ax4.set_title("step epsilon")
        plt.tight_layout()
        plt.savefig(fig_filename, dpi=500)
