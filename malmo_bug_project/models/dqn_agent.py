import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.replay_buffer import ReplayBuffer

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.model = Net(state_dim, action_dim)
        self.target = Net(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer()
        self.gamma = 0.99
        self.batch_size = 32
        self.action_dim = action_dim
        self.epsilon = 1.0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            state = torch.FloatTensor(state[:3]).unsqueeze(0)
            return self.model(state).argmax().item()

    def store_transition(self, s, a, r, s_, d):
        self.buffer.push(s[:3], a, r, s_[:3], d)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_vals = self.model(states).gather(1, actions)
        next_q_vals = self.target(next_states).max(1)[0].unsqueeze(1)
        targets = rewards + self.gamma * next_q_vals * (1 - dones)

        loss = nn.MSELoss()(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_target()

    def _update_target(self):
        self.target.load_state_dict(self.model.state_dict())
