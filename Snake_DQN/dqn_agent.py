# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from model import DQN
from replay_buffer import ReplayBuffer
from config import GAMMA, LR, BATCH_SIZE, BUFFER_SIZE, TARGET_UPDATE, EPS_START, EPS_END, EPS_DECAY

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE)

        self.steps_done = 0
        self.epsilon = EPS_START

    def select_action(self, state):
        self.epsilon = max(EPS_END, EPS_DECAY * self.epsilon)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.policy_net(state).argmax(dim=1).item()

    def push(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)

        q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q_values = reward + GAMMA * next_q_values * (1 - done)

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
