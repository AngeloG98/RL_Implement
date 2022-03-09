import torch
import numpy as np
from torch.distributions import Normal
from network import PG_fc_gaussian, PG_value

class PG_B_agent():
    def __init__(self, policy_layer_size, value_layer_size, gamma, p_lr, v_lr) -> None:
        self.traj_s, self.traj_a, self.traj_r = [], [], []
        self.gamma = gamma

        self.policy_net = PG_fc_gaussian(policy_layer_size).cuda()
        self.value_net = PG_value(value_layer_size).cuda()
        self.p_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=p_lr)
        self.v_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=v_lr)

        self.name = 'PG_baseline'
    
    def reset_traj(self):
        self.traj_s, self.traj_a, self.traj_r = [], [], []

    def store_traj(self, state, action, reward):
        self.traj_s.append(state)
        self.traj_a.append(action)
        self.traj_r.append(reward)

    def get_action(self, state, choose_max = False):
        with torch.no_grad():
            mu, sigma = self.policy_net(torch.tensor(state).unsqueeze(0).float().cuda())
            m = Normal(mu, sigma)
            action = m.sample()
        if choose_max == True:
            action = mu
        # action = action.clamp(-2.0, 2.0)
        return action.item()

    def compute_discount_R(self):  
        T = len(self.traj_r)
        rewards = np.array(self.traj_r)
        discount_R = []
        r = 0
        for t in reversed(range(T)):
            r = self.gamma * r + rewards[t]
            discount_R.append(r)
        discount_R = discount_R[::-1] 
        return discount_R

    def learn(self):
        states = torch.tensor(np.array(self.traj_s)).float().cuda()
        actions = torch.tensor(np.array(self.traj_a)).unsqueeze(1).float().cuda()
        dis_rewards = torch.tensor(np.array(self.compute_discount_R())).float().cuda()

        vfs = self.value_net(states).cuda().view(-1)
        with torch.no_grad():
            advs = dis_rewards - vfs

        mu, sigma = self.policy_net(states)
        m = Normal(mu, sigma)

        p_loss = torch.sum(-m.log_prob(actions).sum(axis=-1) * advs)
        self.p_optimizer.zero_grad()
        p_loss.backward()
        self.p_optimizer.step()

        v_loss = torch.nn.MSELoss()(vfs, dis_rewards)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        return np.array([p_loss.item(), v_loss.item()])

    def save_trained_model(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load_pretrained_model(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))