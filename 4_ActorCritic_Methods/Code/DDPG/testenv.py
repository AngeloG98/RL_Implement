import gym
# env = gym.make('LunarLanderContinuous-v2')
env = gym.make('CarRacing-v1')

state = env.reset()
# for _ in range(100):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.shape[0]
max_action=env.action_space.high
min_action= env.action_space.low
print()