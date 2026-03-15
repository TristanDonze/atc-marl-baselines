import torch
import torch.nn as nn

class LearnedDynamics(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, state_dim)
        )

    def forward(self, state, action):
        tau = torch.cat([state, action], dim=-1)
        delta_state = self.net(tau)
        return state + delta_state

class LearnedCost(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

    def forward(self, tau):
        return self.net(tau).squeeze(-1)