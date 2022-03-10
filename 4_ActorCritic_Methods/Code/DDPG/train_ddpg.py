import gym
import torch
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from ddpg import PPO_agent
from datetime import datetime
TIMESTAMP = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.now())

def train(args):
    env = gym.make(args.env)
    episode = 0
    step = 0
    state = env.reset()
    # while True:

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Proximal Policy Optimization Implementation')
    parser.add_argument('--agent_name', type=str, default='PPO')
    parser.add_argument('--env', type=str, default='Pendulum-v1', help='gym environment name(continuous)')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--lambd', type=float, default=1.0, help='exponential weight discount for gae')
    parser.add_argument('--a_lr', type=float, default=1e-3)
    parser.add_argument('--c_lr', type=float, default=1e-3)
    parser.add_argument('--eps_clip', type=float, default=0.2, help='epsilon clip for ppo')
    parser.add_argument('--num_steps', type=int, default=500, help='number of steps each trajectory(or ppo batch size)')
    parser.add_argument('--iter_times', type=int, default=10, help='iteration times each batch')
    parser.add_argument('--max_steps', type=int, default=9999, help='max of steps for each episode')
    parser.add_argument('--v_loss_coef', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--e_loss_coef', type=float, default=0.001, help='entropy loss coefficient')
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=100)
    args = parser.parse_args()
    train(args)