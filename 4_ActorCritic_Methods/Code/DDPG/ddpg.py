import torch.optim as optim

from utils import ReplayMemory
from network import ActorNet, CriticNet

class DDPG_agent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.95, a_lr=1e-3, v_lr=2e-3, tau=0.05, sync_freq=1, exp_replay_size=1000):
        self.exp_replay_mem = ReplayMemory(exp_replay_size)
        
        self.actor = ActorNet(state_dim, action_dim, max_action).cuda()
        self.actor_target = ActorNet(state_dim, action_dim, max_action).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = CriticNet(state_dim, action_dim).cuda()
        self.critic_target = CriticNet(state_dim, action_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=v_lr)

        self.gamma = gamma
        self.tau = tau
        self.sync_freq = sync_freq

    # def 
