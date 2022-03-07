from email import parser
from turtle import st
import gym
import torch
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from a2c import A2C_agent

def train(args):
    env = gym.make('CartPole-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    layer_sizes = [input_dim, 64, output_dim]
    
    agent = A2C_agent(
        layer_sizes = layer_sizes,
        gamma=args.gamma,
        lr=args.lr
    )






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advantage Actor-Critic Implementation')
    parser.add_argument('--agent_name', type=str, default='A2C')
    parser.add_argument('--env', type=str, default='CartPole-v0', help='gym environment name')
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--lr', type=int, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=512, help='number of steps each trajectory')
    args = parser.parse_args()
    train(args)
    