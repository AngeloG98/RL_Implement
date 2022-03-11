import random
import gym
import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ddpg import DDPG_agent
from datetime import datetime
TIMESTAMP = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.now())

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
    writer = SummaryWriter('4_ActorCritic_Methods/Log/'+TIMESTAMP+'_'+args.agent_name+'_'+args.env+'/')

    episode = 0
    step = 0
    total_step = 0
    reward_sum = 0
    state = env.reset()
    
    while True:
        action = agent.get_action(state, args.noise)
        state_, reward, is_terminal, _ = env.step(action)
        agent.store_memory(state, action, reward, state_, is_terminal)
        state = state_

        if len(agent.mem) > args.batch_size and total_step % args.learn_freq == 0:
            actor_loss, critic_loss = agent.learn(args.batch_size)
        writer.add_scalar("Loss/actor loss", actor_loss, agent.learn_count)
        writer.add_scalar("Loss/critic loss", critic_loss, agent.learn_count)

        step += 1
        total_step += 1
        reward_sum += reward

        if step > args.max_steps:
            is_terminal = True
        if is_terminal:
            # ==========================================================================================
            # print and save 
            writer.add_scalar("Performance/episode step", step, episode)
            writer.add_scalar("Performance/episode reward", reward_sum, episode)
            if episode % args.print_freq == 0 and episode >= 50:
                print("====================================================")
                print("train model: " + agent.name)
                print("train env: " + args.env)
                print("episode: {}".format(episode))
                print("episode step: {}".format(step))
                print("episode reward: {}".format(reward_sum))
                print("episode losses [actor, critic]: {}".format([actor_loss, critic_loss]))
            if episode % args.save_freq == 0 and episode >= 500:
                    model_filename = "4_ActorCritic_Methods/Model/"+agent.name+"_"+env.env.spec.id+"_episode_"+str(episode)
                    agent.save(model_filename)
            # ==========================================================================================
            state = env.reset()
            episode += 1
            step = 0
            reward_sum = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Deterministic Policy Gradient')
    parser.add_argument('--agent_name', type=str, default='PPO')
    parser.add_argument('--env', type=str, default='Pendulum-v1', help='gym environment name(continuous)')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--a_lr', type=float, default=1e-3)
    parser.add_argument('--c_lr', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=0.005, help='soft update parameter for target network')
    parser.add_argument('--sync_freq', type=int, default=1, help='update frequence for target network')
    parser.add_argument('--exp_replay_size', type=int, default=100000, help='experience replay buffer size')
    parser.add_argument('--noise', type=float, default=0.2, help='noise parameter for choosing action(ensure exploration)')
    parser.add_argument('--learn_freq', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=9999, help='max of steps for each episode')
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=500)
    args = parser.parse_args()
    train(args)