import gym
import numpy as np
import matplotlib.pyplot as plt
from atari_wrapper_pytorch import make_atari, wrap_deepmind
from dqn import DQN_agent

configs = {
    "seed": 16,
    "env": "PongNoFrameskip-v4", # use NoFrameskip versions
    "agent": "DQN", # DQN or 

    # agent hyper-parameters
    "gamma": 0.99, # discount factor

    # test hyper-parameters
    "test_episode": 3,
    # keep for agent
    "lr": 1e-4,
    "exp_replay_size": 5000, # experience replay buffer size
    "sync_freq": 1000, # update target network frequence
}

env = make_atari(configs["seed"], configs["env"])
env = wrap_deepmind(env, frame_stack=True, scale=False)
env = gym.wrappers.Monitor(
    env,
    '2_Deep_ValueBased/Other/',
    video_callable=lambda episode_id: episode_id % 1 == 0,
    force=True
)

agent = DQN_agent(
    seed=configs["seed"],
    input_shape=env.observation_space.shape,
    num_actions=env.action_space.n,
    lr=configs["lr"],
    gamma=configs["gamma"],
    sync_freq=configs["sync_freq"],
    exp_replay_size=configs["exp_replay_size"]
)
model_filename = "2_Deep_ValueBased/Model/DQN_Atari_pretrained/"+agent.name+"-DQN_"+configs["env"]+"_episode_300"+".pth"
agent.load_pretrained_model(model_filename)

for episode in range(configs["test_episode"]):
    loss_sum, reward_sum, step, is_terminal = 0, 0, 0, False
    state = env.reset()

    while not is_terminal:
        action, _ = agent.get_action(agent.norm_state(state), 0.0)
        state, reward, is_terminal, info = env.step(action)

        reward_sum += reward
        step += 1
    
    print("====================================================")
    print("test time: {}".format(episode))
    print("episode step: {}".format(step))
    print("episode reward: {}".format(reward_sum))