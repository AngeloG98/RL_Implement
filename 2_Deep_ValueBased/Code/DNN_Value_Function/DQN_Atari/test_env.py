import gym
import numpy as np
from atari_wrapper_pytorch import make_atari, wrap_deepmind

if __name__ == "__main__":
    configs = {
        "seed": 10,
        "env": "PongNoFrameskip-v4",
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
