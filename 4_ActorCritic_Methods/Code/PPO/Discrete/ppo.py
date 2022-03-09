import torch
import numpy as np
from torch.distributions import Categorical
from network import ActorCriticNet

class PPO_agent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lambd=1.0, a_lr=1e-3, c_lr=1e-3, eps_clip=0.1, v_loss_coef=0.5, e_loss_coef=0.001) -> None:
        self.traj = []
        self.gamma = gamma
        self.lambd = lambd
        self.ppo_net = ActorCriticNet(state_dim, action_dim).cuda()
        self.optimizer = torch.optim.Adam([
            {'params': self.ppo_net.actor_net.parameters(), 'lr': a_lr},
            {'params': self.ppo_net.critic_net.parameters(), 'lr': c_lr}
        ])
        self.eps_clip = eps_clip
        self.v_loss_coef = v_loss_coef
        self.e_loss_coef = e_loss_coef

        self.learn_times = 0
        self.name = 'PPO'
    
    def reset_traj(self):
        del self.traj[:]

    def store_traj(self, state, action, reward, log_prob, is_terminal):
        # reshape to torch.Size([1, value_dim])
        state = torch.tensor(state).unsqueeze(0).float().cuda()
        action = action.unsqueeze(0) if action != None else None
        reward = torch.tensor([reward]).unsqueeze(0).float().cuda() if reward != None else None
        log_prob = log_prob.unsqueeze(0) if log_prob != None else None
        mask = torch.tensor([1-is_terminal]).unsqueeze(0).float().cuda() if is_terminal != None else None
        self.traj.append((state, action, reward, log_prob, mask))

    def get_value(self, state):
        with torch.no_grad():
            _, value = self.ppo_net(torch.tensor(state).unsqueeze(0).float().cuda())
        return value

    def get_action(self, state, choose_max = False):
        with torch.no_grad():
            probs, value = self.ppo_net(torch.tensor(state).unsqueeze(0).float().cuda())
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
        advantage = torch.zeros(1, 1).cuda()

        for t in reversed(range(traj_len)):
            state, action, reward, log_prob, mask = self.traj[t]
            state_, _, _, _, _ = self.traj[t + 1]
            with torch.no_grad():
                probs, value = self.ppo_net(state)
                _, value_ = self.ppo_net(state_)

            # gae advantage
            delta = reward + value_ * self.gamma * mask - value
            advantage = advantage * self.gamma * self.lambd * mask + delta
            # discount reward
            discount_r = advantage + value

            bs_traj[t] = state, action, discount_r, advantage, log_prob
        
        return map(lambda x: torch.cat(x, 0), zip(*bs_traj))

    def learn(self, batch_data):
        states, actions, discount_rs, advantages, old_log_probs = batch_data
        probs, values = self.ppo_net(states)
        m = Categorical(probs)
        log_probs = m.log_prob(actions.view(-1)).unsqueeze(1)
        entropys = m.entropy()

        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (values - discount_rs).pow(2).mean()
        entropy_loss = entropys.mean()

        # minus entropy_loss here， entropy penalty to regulate the learning rate， avoid fast convergence to local minima with deterministic solutions.
        loss = policy_loss + self.v_loss_coef*value_loss - self.e_loss_coef*entropy_loss 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_times += 1

        return policy_loss.item(), value_loss.item(), entropy_loss.item()

    def save_trained_model(self, filename):
        torch.save(self.ppo_net.state_dict(), filename)

    def load_pretrained_model(self, filename):
        self.ppo_net.load_state_dict(torch.load(filename))