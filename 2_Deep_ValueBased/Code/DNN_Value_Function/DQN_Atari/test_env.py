import gym
from gym import envs
import numpy as np
from atari_wrapper_pytorch import make_atari, wrap_deepmind

env_names = [spec.id for spec in envs.registry.all()] 
for name in sorted(env_names): 
	print(name)

configs = {
    "seed": 10,
    "env": "BreakoutNoFrameskip-v4",
    "agent": "ENV",
    "max_step": 1000000
}

env = make_atari(configs["seed"], configs["env"])
env = wrap_deepmind(env, frame_stack=True, scale=False)
env = gym.wrappers.Monitor(env, '2_Deep_ValueBased/Video/'+configs["agent"]+'/', video_callable=lambda episode_id: episode_id%10==0,force=True)

state = env.reset()
for step in range(configs["max_step"]): 
    action = env.action_space.sample()
    state_, reward, is_terminal, info = env.step(action)
    if is_terminal:
        state = env.reset()


