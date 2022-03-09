import gym
import torch
import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO_agent



def test(args):
    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    
    agent = PPO_agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        gamma=args.gamma,
        lambd=args.lambd,
        a_lr=args.a_lr,
        c_lr=args.c_lr,
        eps_clip=args.eps_clip,
        v_loss_coef=args.v_loss_coef,
        e_loss_coef=args.e_loss_coef
    )
    model_filename = "4_ActorCritic_Methods/Model/"+agent.name+"_"+env.env.spec.id+"_episode_1000.pth"
    agent.load_pretrained_model(model_filename)

    episode = 0
    step = 0
    state = env.reset()

    while True:
        # env.render()
        # time.sleep(0.01)
        action, _, _, _ = agent.get_action(state, choose_max=True)
        state_, reward, is_terminal, _ = env.step(action.item())
        state = state_
        step += 1

        if step >= args.max_steps:
            is_terminal = True
        if is_terminal:
            state = env.reset()
            #################################################### Print and Save ####################################################
            print("====================================================")
            print("test model: " + agent.name)
            print("episode: {}".format(episode))
            print("episode step: {}".format(step))
            ######################################################################################################################
            episode += 1
            step = 0
            if episode > args.test_episodes - 1:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Proximal Policy Optimization Implementation')
    parser.add_argument('--agent_name', type=str, default='PPO')
    parser.add_argument('--env', type=str, default='CartPole-v0', help='gym environment name(discrete)')
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--lambd', type=float, default=1.0, help='exponential weight discount for gae')
    parser.add_argument('--a_lr', type=float, default=1e-3)
    parser.add_argument('--c_lr', type=float, default=1e-3)
    parser.add_argument('--eps_clip', type=float, default=0.2, help='epsilon clip for ppo')
    parser.add_argument('--test_episodes', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=9999, help='max of steps for each episode')
    parser.add_argument('--v_loss_coef', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--e_loss_coef', type=float, default=0.001, help='entropy loss coefficient')
    args = parser.parse_args()
    test(args)