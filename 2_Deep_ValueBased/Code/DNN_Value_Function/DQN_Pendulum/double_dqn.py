import copy
import torch
from model import *
from utils import *

class double_DQN_agent():
    def __init__(self, seed, layer_sizes, lr = 1e-3, gamma = 0.9, sync_freq = 5, exp_replay_size = 256) -> None:
        torch.manual_seed(seed)

        self.gamma = torch.tensor(gamma).float().cuda()

        self.net = DQN_fnn(layer_sizes)
        self.target_net = copy.deepcopy(self.net)
        self.net.cuda()
        self.target_net.cuda()

        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.sync_freq = sync_freq
        self.learn_count = 0
        self.exp_replay_mem = ReplayMemory(exp_replay_size)

        self.name = "double"

    def get_action(self, state, action_space_len, epsilon):
        with torch.no_grad():
            Qp = self.net(torch.from_numpy(state).float().cuda())
        Q, A = torch.max(Qp, axis=0)
        A = A if torch.rand(1, ).item() > epsilon else torch.randint(0, action_space_len, (1,))
        return A.item(), Q

    def get_qvalue_(self, state, batch_size):
        with torch.no_grad():
            q_target_net = self.target_net(state)
            q_net = self.net(state)
        _, action_max = torch.max(q_net, axis=1)
        q_max_ = q_target_net.gather(-1, action_max.view(batch_size, 1)).view(-1)
        return q_max_

    def store_memory(self, *exp):
        self.exp_replay_mem.push_pop(*exp)

    def learn(self, batch_size):
        batch = self.exp_replay_mem.sample(batch_size)
        state = torch.tensor(np.float32(batch.state)).float().cuda()
        action = torch.LongTensor(batch.action).cuda()
        reward = torch.tensor(batch.reward).float().cuda()
        state_ = torch.tensor(np.float32(batch.state_)).float().cuda()
        is_terminal = torch.tensor(batch.is_terminal).float().cuda()

        if self.learn_count % self.sync_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        q_values = self.net(state)
        q_value = q_values.gather(-1, action.view(batch_size, 1)).view(-1)

        max_q_value_ = self.get_qvalue_(state_, batch_size)
        q_target = reward + self.gamma * max_q_value_ * (1 - is_terminal)

        loss = self.loss_func(q_value.view(batch_size, 1), q_target.view(batch_size, 1))
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.learn_count += 1

        return loss.item() 

    def save_trained_model(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load_pretrained_model(self, filename):
        self.net.load_state_dict(torch.load(filename))