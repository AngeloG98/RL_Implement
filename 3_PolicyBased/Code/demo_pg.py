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
    'Type': ['trajectory', 'total'],
    'test_episode': 3,
    'print_freq': 20,
}

agent = PG_agent(
    configs['layer_sizes'],
    configs['gamma'],
    configs['lr']
)
model_filename = "3_PolicyBased/Model/"+agent.name+"_"+env.env.spec.id+"_episode_1500.pth"
agent.load_pretrained_model(model_filename)

for episode in range(configs['test_episode']):
    reward_sum, step, is_terminal = 0, 0, False
    state = env.reset()

    while not is_terminal:
        # env.render()
        action = agent.get_action(state, choose_max=True)
        state_, reward, is_terminal, info = env.step(action)
        # reward = reward_func(env, state_)

        state = state_
        reward_sum += reward
        step += 1

    print("====================================================")
    print("train model: " + agent.name)
    print("episode: {}".format(episode))
    print("episode step: {}".format(step))
    print("episode reward: {}".format(reward_sum))