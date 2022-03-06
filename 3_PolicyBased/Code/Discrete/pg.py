import torch
import numpy as np
from torch.distributions import Categorical
from network import PG_fc

class PG_agent():
    def __init__(self, layer_sizes, gamma, lr) -> None:
        self.traj_s, self.traj_a, self.traj_r = [], [], []
        self.gamma = gamma

        self.policy_net = PG_fc(layer_sizes).cuda()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.name = 'PG'
    
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

    def compute_discount_R(self, Type):  
        T = len(self.traj_r)
        rewards = np.array(self.traj_r)
        discount_R = []
        # total discount reward and future discount reward
        if Type == 'total':
            for t in range(T):
                r = np.sum(rewards[t:] * (self.gamma**np.array(range(t, T))))
                discount_R.append(r)
        else:
            r = 0
            for t in reversed(range(T)):
                r = self.gamma * r + rewards[t]
                discount_R.append(r)
            discount_R = discount_R[::-1] 
        return discount_R

    def learn(self, Type):
        states = torch.tensor(np.array(self.traj_s)).float().cuda()
        actions = torch.tensor(np.array(self.traj_a)).float().cuda()
        dis_rewards = torch.tensor(np.array(self.compute_discount_R(Type[1]))).float().cuda()

        if Type[0] == 'trajectory':
            probs = self.policy_net(states)
            m = Categorical(probs)
            aaa = -m.log_prob(actions) * dis_rewards
            # loss = torch.mean(-m.log_prob(actions) * dis_rewards)
            loss = torch.sum(-m.log_prob(actions) * dis_rewards)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            for i in range(len(self.traj_s)):
                probs = self.policy_net(states[i].unsqueeze(0))
                m = Categorical(probs)
                loss = -m.log_prob(actions[i]) * dis_rewards[i]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    def save_trained_model(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load_pretrained_model(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))