import random
import torch.optim as optim
import matplotlib.pyplot as plt
from common import *
from model import *
from torch import FloatTensor, LongTensor

class DQN():
    def __init__(self, env, gamma = 0.9, epsilon = 0.1, buffer_size = 2000, device = 'cuda') -> None:
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        self.action_space = [i for i in range(env.action_space.n)]
        self.replay_memory = ReplayMemory(buffer_size)

        self.net = Net(env.observation_space.shape, env.action_space.n).to(device)
        self.target_net = Net(env.observation_space.shape, env.action_space.n).to(device)
        self.loss_func = nn.MSELoss()
        self.device = device
          
    def epsilon_greed(self, state):
        if random.uniform(0, 1.0) > self.epsilon:
            state_tensor = torch.FloatTensor(np.float32(state)).to(self.device)
            a = self.net.forward(state_tensor)
            action = self.net.forward(state_tensor).max(0)[1].item()
        else:
            action = random.choice(self.action_space)
        return action

    def greed(self, state):
        state_tensor = torch.FloatTensor(np.float32(state)).to(self.device)
        return self.net.forward(state_tensor).max(0)[1].item()

    def learn(self, optimizer, batch_size):
        batch = self.replay_memory.sample(batch_size)

        state = Variable(FloatTensor(np.float32(batch.state))).to(self.device)
        action = Variable(LongTensor(batch.action)).to(self.device)
        reward = Variable(FloatTensor(batch.reward)).to(self.device)
        state_ = Variable(FloatTensor(np.float32(batch.state_))).to(self.device)
        is_terminal = Variable(FloatTensor(batch.is_terminal)).to(self.device)

        q_values = self.net(state)
        q_values_ = self.target_net(state_).detach()

        q_value = q_values.gather(1, action.unsqueeze(-1))
        max_q_value_ = q_values_.max(1)[0]
        q_target = reward + self.gamma * max_q_value_
        # q_target = reward + self.gamma * max_q_value_ * (1 - is_terminal)

        loss = self.loss_func(q_target.view(batch_size, 1), q_value.view(batch_size, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()      

        return loss.item()  
    
    def train(self, episodes = 1000, max_step = 1000, lr = 0.01, batch_size = 32, target_net_update = 10):
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        learn_count = 0
        episode_reward_list = []
        loss_list = []
        for episode in range(episodes+1):
            state = self.env.reset()

            episode_reward = 0
            for step in range(max_step):
                if episode > 300:
                    # self.env.render()
                    pass

                action = self.epsilon_greed(state)
                state_, reward, is_terminal, info = self.env.step(action)

                x, x_dot, theta, theta_dot = state_

                # reward function for "CartPole-v0"
                if str(self.env.spec.id) == "CartPole-v0":
                    r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                    r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                    reward = r1 + r2
                episode_reward += reward

                self.replay_memory.push_pop(state, action, reward, state_, is_terminal)


                if len(self.replay_memory) >= self.replay_memory.capacity:
                    learn_count += 1
                    if learn_count % target_net_update == 0:
                        self.target_net.load_state_dict(self.net.state_dict())
                    loss = self.learn(optimizer, batch_size)
                    loss_list.append(loss)
                    
                if is_terminal or step == max_step - 1:
                    print("use_step:{}, episode:{}, reward_sum: {:.2f}, learn_count:{}".format(step, episode, episode_reward, learn_count))
                    break

                state = state_

            if episode % 400 == 0 and episode != 0:
                self.save_net(episode)
            episode_reward_list.append(episode_reward)
        return episode_reward_list, loss_list

    def save_net(self, episode):
        torch.save(self.net.state_dict(), "2_Deep_ValueBased/Model/DQN-" + str(self.env.spec.id) + "_episode_" + str(episode)+".pkl")

    def forward(self, test_step = 10000, filename = None):
        if filename != None:
            self.net.load_state_dict(torch.load(filename))
        
        self.env.reset()
        state = self.env.state
        for step in range(test_step):
            self.env.render()
            action = self.greed(state)
            state, reward, is_terminal, info = self.env.step(action)
            if is_terminal:
                print("use_step:{}".format(step))
                break

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using " + device.type + "...")
    # parameters
    episodes = 10000
    max_step = 10000
    lr = 0.01
    gamma = 0.9
    epsilon = 0.1
    batch_size = 32
    replay_buffer = 200
    target_net_update = 100
    # environment
    env = gym.make("CartPole-v0")
    env = env.unwrapped
    # model
    dqn = DQN(env, gamma, epsilon, replay_buffer, device)
    episode_reward_list, loss_list = dqn.train(episodes, max_step, lr, batch_size, target_net_update)
    # for _ in range(10):
    #     dqn.forward(test_step = 10000) 
    plt.subplot(211)
    plt.plot(loss_list)
    plt.subplot(212)
    plt.plot(episode_reward_list)
    plt.show()
    # test
    # dqn.forward(test_step = 10000, filename = '2_Deep_ValueBased/Model/DQN-CartPole-v0_episode_100.pkl')
    # dqn.forward(test_step = 10000, filename = "2_Deep_ValueBased/Model/DQN-.pkl")
