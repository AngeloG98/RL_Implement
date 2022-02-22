import gym
import time
from numpy import random


class GridWorldEnv(gym.Env):
    def __init__(self):
        self.viewer = None
        # state space
        self.states = [1, 2, 3, 4, 5,
                       6, 7, 8, 9, 10,
                       11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20,
                       21, 22, 23, 24, 25]
        # action space
        self.actions = ['n', 'e', 's', 'w']

        # terminal reward function
        # maze
        self.terminal_rewards = dict()
        self.terminal_rewards[8] = 10
        self.terminal_rewards[7] = -10
        self.terminal_rewards[13] = -10
        self.terminal_rewards[15] = -10
        self.terminal_rewards[17] = -10
        self.terminal_rewards[24] = -10

        # cliff
        # self.terminal_rewards[25] = 10
        # self.terminal_rewards[17] = -10
        # self.terminal_rewards[18] = -10
        # self.terminal_rewards[19] = -10
        # self.terminal_rewards[22] = -10
        # self.terminal_rewards[23] = -10
        # self.terminal_rewards[24] = -10

        self.init_state_action()

    def cango_n(self, row, col):
        row = row - 1
        return 0 <= row <= 4

    def cango_s(self, row, col):
        row = row + 1
        return 0 <= row <= 4

    def cango_w(self, row, col):
        col = col - 1
        return 0 <= col <= 4

    def cango_e(self, row, col):
        col = col + 1
        return 0 <= col <= 4
    
    def state2rowcol(self, state):
        row = (state - 1) // 5
        col = (state - 1) %  5
        return row, col

    def rowcol2state(self, row, col):
        return 5*row + col + 1

    def init_state_action(self):
        # state transition, state-action pair, deterministic here
        self.states_actions = dict()
        for state in self.states:
            if state not in self.terminal_rewards:
                self.states_actions[state] = dict()
                row, col = self.state2rowcol(state)
                if self.cango_n(row, col):
                    self.states_actions[state]['n'] = self.rowcol2state(row-1, col)
                else:
                    self.states_actions[state]['n'] = state
                if self.cango_e(row, col):
                    self.states_actions[state]['e'] = self.rowcol2state(row, col+1)
                else:
                    self.states_actions[state]['e'] = state
                if self.cango_s(row, col):
                    self.states_actions[state]['s'] = self.rowcol2state(row+1, col)
                else:
                    self.states_actions[state]['s'] = state
                if self.cango_w(row, col):
                    self.states_actions[state]['w'] = self.rowcol2state(row, col-1)
                else:
                    self.states_actions[state]['w'] = state

    def set_state(self, state):
        self.state = state

    def step(self, action):
        # current state
        state = self.state
        # 'state-action' transition
        if (state in self.states_actions) and (action in self.states_actions[state]):
            next_state = self.states_actions[state][action]
        # else stay!
        else:
            next_state = state
        self.state = next_state

        is_terminal = False
        # if is_terminal = True
        if self.state in self.terminal_rewards:
            r = self.terminal_rewards[self.state]
            is_terminal = True
        else:
            r = -1 # none terminal reward -1
        return next_state, r, is_terminal, {}

    def reset(self):
        self.state = self.states[int(random.random() * (len(self.states) ))]
        while self.state in self.terminal_rewards:
            self.state = self.states[int(random.random() * (len(self.states) - 1))]
        return self.state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering
        width = 60
        height = 60
        edge_x = 0
        edge_y = 0
        if self.viewer is None:
            self.viewer = rendering.Viewer(300, 300)
        # terminals
        for key, value in self.terminal_rewards.items():
            row, col = self.state2rowcol(key)
            if value > 0:
                self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                        color=(0, 0, 1.0)).add_attr(rendering.Transform((edge_x + width * col, edge_y + height * (4-row))))
            else:
                self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                        color=(0, 0, 0)).add_attr(rendering.Transform((edge_x + width * col, edge_y + height * (4-row))))
        # line
        for i in range(1, 7):
            self.viewer.draw_line((edge_x, edge_y + height * (i - 1)), (edge_x + 5 * width, edge_y + height * (i - 1)))  
            self.viewer.draw_line((edge_x + width * (i - 1), edge_y + height * 0),
                                  (edge_x + width * (i - 1), edge_y + height * 5))  
        # dot
        self.x = [edge_x + width * 0.5, edge_x + width * 1.5, edge_x + width * 2.5, edge_x + width * 3.5, edge_x + width * 4.5] * 5
        self.y = [edge_y + height * 4.5, edge_y + height * 4.5, edge_y + height * 4.5, edge_y + height * 4.5, edge_y + height * 4.5,
                  edge_y + height * 3.5, edge_y + height * 3.5, edge_y + height * 3.5, edge_y + height * 3.5, edge_y + height * 3.5,
                  edge_y + height * 2.5, edge_y + height * 2.5, edge_y + height * 2.5, edge_y + height * 2.5, edge_y + height * 2.5,
                  edge_y + height * 1.5, edge_y + height * 1.5, edge_y + height * 1.5, edge_y + height * 1.5, edge_y + height * 1.5,
                  edge_y + height * 0.5, edge_y + height * 0.5, edge_y + height * 0.5, edge_y + height * 0.5, edge_y + height * 0.5]
        self.viewer.draw_circle(18, color=(1.0, 0.0, 0.0)).add_attr(
            rendering.Transform(translation=(self.x[self.state - 1], self.y[self.state - 1])))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')




