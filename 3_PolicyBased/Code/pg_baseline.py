import torch
import numpy as np
from torch.distributions import Categorical
from network import PG_fc, PG_value

class PG_B_agent():
    def __init__(self, policy_layer_size, value_layer_size, gamma, p_lr, v_lr) -> None:
        self.traj_s, self.traj_a, self.traj_r = [], [], []
        self.gamma = gamma

        self.policy_net = PG_fc(policy_layer_size).cuda()
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
            probs = self.policy_net(torch.tensor(state).unsqueeze(0).float().cuda())
            m = Categorical(probs)
            action = m.sample()
        if choose_max == True:
            action = torch.argmax(probs,dim=1)
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

    def learn(self, Type=None):
        states = torch.tensor(np.array(self.traj_s)).float().cuda()
        actions = torch.tensor(np.array(self.traj_a)).float().cuda()
        dis_rewards = torch.tensor(np.array(self.compute_discount_R())).float().cuda()

        vfs = self.value_net(states).cuda()
        with torch.no_grad():
            advs = dis_rewards - vfs

        probs = self.policy_net(states)
        m = Categorical(probs)

        p_loss = torch.sum(-m.log_prob(actions) * advs)
        self.p_optimizer.zero_grad()
        p_loss.backward()
        self.p_optimizer.step()

        v_loss = torch.nn.MSELoss()(vfs, dis_rewards)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        return p_loss.item(), v_loss.item()

    def save_trained_model(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load_pretrained_model(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))