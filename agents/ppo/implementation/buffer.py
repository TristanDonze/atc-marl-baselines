import torch

class RolloutBuffer:
    def __init__(self, n_steps, num_envs, device):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.device = device
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def add(self, state, action, reward, done, value, log_prob):
        self.states.append(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        self.actions.append(torch.as_tensor(action, device=self.device))
        self.rewards.append(torch.as_tensor(reward, dtype=torch.float32, device=self.device))
        self.dones.append(torch.as_tensor(done, dtype=torch.float32, device=self.device))
        self.values.append(torch.as_tensor(value, dtype=torch.float32, device=self.device))
        self.log_probs.append(torch.as_tensor(log_prob, dtype=torch.float32, device=self.device))

    def get_tensors(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)
        return states, actions, rewards, dones, values, log_probs