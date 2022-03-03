import copy
import torch
import torch.nn.functional as F
import numpy as np
from model import DQN_conv_duel
from utils import Prior_ReplayMemory

class D3QN_PER_agent():
    def __init__(self, seed, input_shape, num_actions, lr = 1e-3, gamma = 0.9, sync_freq = 5, exp_replay_size = 256) -> None:
        torch.manual_seed(seed)

        self.gamma = torch.tensor(gamma).float().cuda()

        self.net = DQN_conv_duel(input_shape, num_actions)
        self.target_net = copy.deepcopy(self.net)
        self.net.cuda()
        self.target_net.cuda()

        self.loss_func = torch.nn.MSELoss()
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=lr)

        self.sync_freq = sync_freq
        self.learn_count = 0
        self.exp_replay_mem = Prior_ReplayMemory(exp_replay_size)

        self.name = "d3qn_per"
        self.input_shape = input_shape
        self.num_actions = num_actions

    def norm_state(self, state): 
        return np.float32(state)/255 # to numpy and normalize

    def get_action(self, state, epsilon):
        with torch.no_grad():
            Qp = self.net(torch.from_numpy(state).float().unsqueeze(0).cuda()) # unsqueeze batch axis = 1
        Q, A = torch.max(Qp, axis=1) # batch axis
        A = A if torch.rand(1, ).item() > epsilon else torch.randint(0, self.num_actions, (1,))
        return A.item(), Q

    def get_qvalue_(self, state):
        with torch.no_grad():
            q_target_net = self.target_net(state)
            q_net = self.net(state)
        _, action_max = torch.max(q_net, axis=1)
        q_max_ = q_target_net.gather(-1, action_max.unsqueeze(1)).view(-1)
        return q_max_

    def store_memory(self, state, action, reward, state_, is_terminal):
        self.exp_replay_mem.push(state, action, reward, state_, is_terminal)

    def learn(self, batch_size):
        idxs, state, action, reward, state_, is_terminal = self.exp_replay_mem.sample(batch_size)
        state = torch.tensor(np.float32(state)).float().cuda()
        action = torch.LongTensor(action).cuda()
        reward = torch.tensor(reward).float().cuda()
        state_ = torch.tensor(np.float32(state_)).float().cuda()
        is_terminal = torch.tensor(is_terminal).float().cuda()

        if self.learn_count % self.sync_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        q_values = self.net(state)
        q_value = q_values.gather(-1, action.view(batch_size, 1)).view(-1)

        max_q_value_ = self.get_qvalue_(state_)
        q_target = reward + self.gamma * max_q_value_ * (1 - is_terminal)

        errors = (q_value - q_target).detach().cpu().squeeze().tolist()
        self.exp_replay_mem.update(idxs, errors)

        loss = self.loss_func(q_value, q_target)
        # loss = F.smooth_l1_loss(q_value, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_count += 1

        return loss.item() 

    def save_trained_model(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load_pretrained_model(self, filename):
        self.net.load_state_dict(torch.load(filename))