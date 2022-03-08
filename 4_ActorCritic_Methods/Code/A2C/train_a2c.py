import gym
import torch
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from a2c import A2C_agent
from datetime import datetime
TIMESTAMP = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.now())

def train(args):
    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    layer_size = [input_dim, 64, output_dim]
    
    agent = A2C_agent(
        layer_size = layer_size,
        gamma=args.gamma,
        lambd=args.lambd,
        lr=args.lr,
        v_loss_coef=args.v_loss_coef,
        e_loss_coef=args.e_loss_coef
    )

    writer = SummaryWriter('4_ActorCritic_Methods/Log/'+TIMESTAMP+'_'+args.agent_name+'/')
    
    episode = 0
    step = 0
    state = env.reset()
    writer.add_graph(agent.a2c_net, torch.tensor(state).unsqueeze(0).float().cuda())
    
    while True:
        for i in range(args.num_steps):
            action, log_prob, entropy, value = agent.get_action(state)
            
            state_, reward, is_terminal, info = env.step(action.item())
            agent.store_traj(state, value, action, reward, log_prob, entropy, is_terminal)
            state = state_
            step += 1
            
            if step >= args.max_steps:
                is_terminal = True
            if is_terminal:
                state = env.reset()
                writer.add_scalar("Performance/episode step", step, episode)
                if episode % args.print_freq == 0 and episode >= 10:
                    print("====================================================")
                    print("train model: " + agent.name)
                    print("episode: {}".format(episode))
                    print("episode step: {}".format(step))
                    print("episode losses [policy, value, entropy]: {}".format([policy_loss, value_loss, entropy_loss]))
                if episode % args.save_freq == 0 and episode >= 1500:
                    model_filename = "4_ActorCritic_Methods/Model/"+agent.name+"_"+env.env.spec.id+"_episode_"+str(episode)+".pth"
                    agent.save_trained_model(model_filename)
                episode += 1
                step = 0
        
        # collect next state value for bootstrap
        value = agent.get_value(state)
        agent.store_traj(state, value, None, None, None, None, None)
        # TD bootstrap
        bs_traj = agent.bootstrap_traj()
        # learn
        policy_loss, value_loss, entropy_loss = agent.learn(bs_traj)
        # empty trajectory
        agent.reset_traj()
        writer.add_scalar("Loss/policy loss", policy_loss, episode)
        writer.add_scalar("Loss/value loss", value_loss, episode)
        writer.add_scalar("Loss/entropy loss", entropy_loss, episode)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advantage Actor-Critic Implementation')
    parser.add_argument('--agent_name', type=str, default='A2C')
    parser.add_argument('--env', type=str, default='CartPole-v0', help='gym environment name(discrete)')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--lambd', type=float, default=1.0, help='exponential weight discount for gae')
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--num_steps', type=int, default=100, help='number of steps each trajectory(or batch size)')
    parser.add_argument('--max_steps', type=int, default=9999, help='max of steps for each episode')
    parser.add_argument('--v_loss_coef', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--e_loss_coef', type=float, default=0.001, help='entropy loss coefficient')
    parser.add_argument('--print_freq', type=float, default=5)
    parser.add_argument('--save_freq', type=float, default=100)
    args = parser.parse_args()
    train(args)
    