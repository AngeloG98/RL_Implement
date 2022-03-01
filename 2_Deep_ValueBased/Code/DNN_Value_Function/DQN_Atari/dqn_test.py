import copy
import torch
from model import *
from utils import *

class DQN_agent1():
    def __init__(self, seed, in_channels, action_space, lr = 1e-3, gamma = 0.9, sync_freq = 5, exp_replay_size = 256) -> None:
        torch.manual_seed(seed)

        self.action_space = action_space
        self.gamma = torch.tensor(gamma).float().cuda()

        self.net = DQN_conv(in_channels = in_channels, num_actions = action_space.n)
        self.target_net = copy.deepcopy(self.net)
        self.net.cuda()
        self.target_net.cuda()

        self.loss_func = torch.nn.MSELoss()
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(),lr=lr, eps=0.001, alpha=0.95)

        self.sync_freq = sync_freq
        self.learn_count = 0
        self.memory_buffer = Memory_Buffer(exp_replay_size)

        self.name = "natural"

    def get_state(self, lazyframe):
        state = lazyframe._force().transpose(2,0,1)[None]/255
        return state

    def get_action(self, state, epsilon):
        with torch.no_grad():
            # Qp = self.net(torch.from_numpy(state).float().cuda())
            Qp = self.net(state)
        Q, A = torch.max(Qp, axis=1)
        A = A if torch.rand(1, ).item() > epsilon else torch.randint(0, self.action_space.n, (1, ))
        return A.item(), Q

    def get_qvalue_(self, state):
        with torch.no_grad():
            q_values_ = self.target_net(state)
        q_max, _ = torch.max(q_values_, axis=1)
        return q_max

    def observe(self, lazyframe):
        # from Lazy frame to tensor
        state =  torch.from_numpy(lazyframe._force().transpose(2,0,1)[None]/255).float()
        return state.cuda()

    def sample_from_buffer(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.memory_buffer.size() - 1)
            data = self.memory_buffer.buffer[idx]
            frame, action, reward, next_frame, done= data
            states.append(self.observe(frame))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.observe(next_frame))
            dones.append(done)
        return torch.cat(states), actions, rewards, torch.cat(next_states), dones

    def learn(self, batch_size):
        state, action, reward, state_, is_terminal = self.sample_from_buffer(batch_size)
        action = torch.LongTensor(action).cuda()
        is_terminal = torch.tensor(is_terminal).float().cuda()
        reward = torch.tensor(reward).float().cuda()

        if self.learn_count % self.sync_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        q_values = self.net(state)
        q_value = q_values.gather(-1, action.view(batch_size, 1)).view(-1)

        max_q_value_ = self.get_qvalue_(state_)
        q_target = reward + self.gamma * max_q_value_ * (1 - is_terminal)

        # loss = self.loss_func(q_value, q_target)
        loss = F.smooth_l1_loss(q_value, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_count += 1

        return loss.item() 

    def save_trained_model(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load_pretrained_model(self, filename):
        self.net.load_state_dict(torch.load(filename))