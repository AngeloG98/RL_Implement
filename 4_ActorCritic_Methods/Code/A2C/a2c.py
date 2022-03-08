import torch
import numpy as np
from torch.distributions import Categorical
from network import ActorCriticNet

class A2C_agent():
    def __init__(self, layer_size, gamma = 0.99, lambd = 1.0, lr = 1e-3, v_loss_coef = 0.5, e_loss_coef = 0.001) -> None:
        self.traj = []
        self.gamma = gamma
        self.lambd = lambd
        self.a2c_net = ActorCriticNet(layer_size).cuda()
        self.optimizer = torch.optim.Adam(self.a2c_net.parameters(), lr=lr)
        self.v_loss_coef = v_loss_coef
        self.e_loss_coef = e_loss_coef

        self.name = 'A2C'
    
    def reset_traj(self):
        self.traj = []

    def store_traj(self, state, value, action, reward, log_prob, entropy, is_terminal):
        # reshape to torch.Size([1, value_dim])
        state = torch.tensor(state).unsqueeze(0).float().cuda()
        action = action.unsqueeze(0) if action != None else None
        reward = torch.tensor([reward]).unsqueeze(0).float().cuda() if reward != None else None
        log_prob = log_prob.unsqueeze(0) if log_prob != None else None
        entropy = entropy.unsqueeze(0) if entropy != None else None
        mask = torch.tensor([1-is_terminal]).unsqueeze(0).float().cuda() if is_terminal != None else None
        self.traj.append((state, value, action, reward, log_prob, entropy, mask))

    def get_value(self, state):
        # with torch.no_grad():
        _, value = self.a2c_net(torch.tensor(state).unsqueeze(0).float().cuda())
        return value

    def get_action(self, state, choose_max = False):
        # with torch.no_grad():
        probs, value = self.a2c_net(torch.tensor(state).unsqueeze(0).float().cuda())
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy()
        if choose_max == True:
            action = torch.argmax(probs,dim=1)
        return action, log_prob, entropy, value

    def bootstrap_traj(self):
        traj_len = len(self.traj) - 1 # we collect next state and value for bootstrap earlier, so here should be len()-1
        bs_traj = [None] * traj_len
        _, discount_r, _, _, _, _, _ = self.traj[-1]
        advantage = torch.zeros(1, 1).cuda()

        for t in reversed(range(traj_len)):
            state, value, action, reward, log_prob, entropy, mask = self.traj[t]
            state_, value_, _, _, _, _, _ = self.traj[t + 1]
            
            # discount reward
            discount_r = reward + discount_r * self.gamma * mask
            # gae advantage
            delta = reward + value_ * self.gamma * mask - value
            advantage = advantage * self.gamma * self.lambd * mask + delta

            bs_traj[t] = state, value, action, discount_r, advantage, log_prob, entropy
        
        return map(lambda x: torch.cat(x, 0), zip(*bs_traj))

    def learn(self, bs_traj):
        # compute log_probs\values\entropys, if use 'with torch.no_grad()' when collecting
        # states, _, actions, discount_rs, advantages, _, _ = bs_traj
        # probs, values = self.a2c_net(states)
        # m = Categorical(probs)
        # log_probs = m.log_prob(actions.view(-1))
        # entropys = m.entropy()
        # policy_loss = (-log_probs * advantages.view(-1)).sum() # sum or mean
        # value_loss = (values - discount_rs).pow(2).sum()
        
        # bootstrap with grad
        _, values, _, discount_rs, advantages, log_probs, entropys = bs_traj
        policy_loss = (-log_probs * advantages.detach()).mean()
        value_loss = (values - discount_rs.detach()).pow(2).mean()

        entropy_loss = entropys.mean()
        loss = policy_loss + self.v_loss_coef*value_loss + self.e_loss_coef*entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy_loss.item()

    def save_trained_model(self, filename):
        torch.save(self.a2c_net.state_dict(), filename)

    def load_pretrained_model(self, filename):
        self.a2c_net.load_state_dict(torch.load(filename))