import random
import gym
import torch
import argparse
import time
from ddpg import DDPG_agent

def train(args):
    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    agent = DDPG_agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high,
        gamma=args.gamma,
        a_lr=args.a_lr,
        c_lr=args.c_lr,
        tau=args.tau,
        sync_freq=args.sync_freq,
        exp_replay_size=args.exp_replay_size
    )
    model_filename = "4_ActorCritic_Methods/Model/"+agent.name+"_"+env.env.spec.id+"_episode_1600"+"_"
    agent.load(model_filename)

    episode = 0
    step = 0
    total_step = 0
    reward_sum = 0
    state = env.reset()
    
    while True:
        env.render()
        time.sleep(0.01)
        action = agent.get_action(state)
        state_, reward, is_terminal, _ = env.step(action)
        reward = reward/10
        state = state_

        step += 1
        total_step += 1
        reward_sum += reward

        if step > args.max_steps:
            is_terminal = True
        if is_terminal:
            # ==========================================================================================
            # print and save 
            print("======================================================================")
            print("test model: " + agent.name)
            print("train env: " + args.env)
            print("episode: {}".format(episode))
            print("episode step: {}".format(step))
            print("episode reward: {}".format(reward_sum))
            # ==========================================================================================
            state = env.reset()
            episode += 1
            step = 0
            reward_sum = 0
            if episode > args.test_episodes - 1:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Deterministic Policy Gradient')
    parser.add_argument('--agent_name', type=str, default='DDPG')
    parser.add_argument('--env', type=str, default='Pendulum-v1', help='gym environment name(continuous)')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--a_lr', type=float, default=1e-3)
    parser.add_argument('--c_lr', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=0.005, help='soft update parameter for target network')
    parser.add_argument('--sync_freq', type=int, default=1, help='update frequence for target network')
    parser.add_argument('--exp_replay_size', type=int, default=100000, help='experience replay buffer size')
    parser.add_argument('--max_steps', type=int, default=999, help='max of steps for each episode')
    parser.add_argument('--test_episodes', type=int, default=3)
    args = parser.parse_args()
    train(args)