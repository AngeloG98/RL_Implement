from csv import QUOTE_NONE
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import ReplayMemory
from network import ActorNet, CriticNet

class DDPG_agent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, a_lr=1e-3, c_lr=2e-3, tau=0.005, sync_freq=1, exp_replay_size=10000):
        self.mem = ReplayMemory(exp_replay_size)
        
        self.actor = ActorNet(state_dim, action_dim, max_action[0]).cuda() # if max_aciton dim > 1 ?
        self.actor_target = ActorNet(state_dim, action_dim, max_action[0]).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = CriticNet(state_dim, action_dim).cuda()
        self.critic_target = CriticNet(state_dim, action_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=c_lr)

        self.gamma = gamma
        self.tau = tau
        self.sync_freq = sync_freq
        self.learn_count = 0

        self.name = 'DDPG'

    def store_memory(self, *exp):
        self.mem.push_pop(exp)
    
    def get_action(self, state, noise=0.0):
        with torch.no_grad():
            action = self.actor(torch.tensor(state).float().cuda())
        # add guassian noise for exploration
        mean = torch.full(action.shape, 0.0)
        std = torch.full(action.shape, noise)
        action += torch.normal(mean, std).cuda()
        return action.cpu().numpy()
    
    def learn(self, batch_size):
        batch = self.mem.sample(batch_size)
        states = torch.tensor(np.float32(batch.state)).float().cuda()
        actions = torch.tensor(np.float32(batch.action)).float().cuda()
        rewards = torch.tensor(batch.reward).unsqueeze(1).float().cuda()
        states_ = torch.tensor(np.float32(batch.state_)).float().cuda()
        is_terminals = torch.tensor(batch.is_terminal).unsqueeze(1).float().cuda()

        with torch.no_grad():
            q_value_ = self.critic_target(states_, self.actor_target(states_))
            q_target = rewards + self.gamma * q_value_ * (1 - is_terminals)
        q_value = self.critic(states, actions)
        critic_loss = F.mse_loss(q_target, q_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learn_count % self.sync_freq == 0:
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

        self.learn_count += 1

        return actor_loss.item(), critic_loss.item()


    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + 'actor.pth')
        torch.save(self.critic.state_dict(), filename + 'critic.pth')

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + 'actor.pth'))
        self.critic.load_state_dict(torch.load(filename + 'critic.pth'))