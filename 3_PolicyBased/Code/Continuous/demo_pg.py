import gym
import numpy as np
import time
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
    'test_episode': 3
}

agent = PG_B_agent(
    configs['policy_layer_size'],
    configs['value_layer_size'],
    configs['gamma'],
    configs['p_lr'],
    configs['v_lr']
)
model_filename = "3_PolicyBased/Model/"+agent.name+"_"+env.env.spec.id+"_episode_6600.pth"
agent.load_pretrained_model(model_filename)

for episode in range(configs['test_episode']):
    reward_sum, step, is_terminal = 0, 0, False
    state = env.reset()

    while not is_terminal:
        env.render()
        time.sleep(0.01)
        action = agent.get_action(state, choose_max=True)
        state_, reward, is_terminal, info = env.step([action])
        reward = reward/10

        state = state_
        reward_sum += reward
        step += 1

    print("====================================================")
    print("train model: " + agent.name)
    print("episode: {}".format(episode))
    print("episode step: {}".format(step))
    print("episode reward: {}".format(reward_sum))