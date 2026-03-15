import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
import random
import aim
from mpc import mpc
import traceback
from gym_air_traffic.envs.air_traffic_env import AirTrafficEnv
from src.networks import LearnedDynamics, LearnedCost

warnings.filterwarnings("ignore")

class WrappedDynamics(nn.Module):
    def __init__(self, base_model, s_mean, s_std):
        super().__init__()
        self.base = base_model
        self.s_mean = s_mean
        self.s_std = s_std

    def forward(self, state, action):
        norm_s = (state - self.s_mean) / self.s_std
        norm_next_s = self.base(norm_s, action)
        next_s = (norm_next_s * self.s_std) + self.s_mean
        return next_s

class WrappedCost(nn.Module):
    def __init__(self, base_model, state_dim, s_mean, s_std, c_mean, c_std):
        super().__init__()
        self.base = base_model
        self.state_dim = state_dim
        self.s_mean = s_mean
        self.s_std = s_std
        self.c_mean = c_mean.squeeze(-1)
        self.c_std = c_std.squeeze(-1)

    def forward(self, tau):
        state = tau[..., :self.state_dim]
        action = tau[..., self.state_dim:]
        
        norm_s = (state - self.s_mean) / self.s_std
        norm_tau = torch.cat([norm_s, action], dim=-1)
        
        norm_c = self.base(norm_tau)
        
        action_penalty = 0.1 * (action ** 2).sum(dim=-1)
        total_cost = norm_c + action_penalty
        
        if tau.dim() == 3:
            return total_cost.sum(dim=0)
        return total_cost

class PrioritizedBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []

    def add(self, state, action, next_state, cost, error):
        priority = (error + 1e-6) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, next_state, cost))
            self.priorities.append(priority)
        else:
            idx = random.randint(0, self.capacity - 1)
            self.buffer[idx] = (state, action, next_state, cost)
            self.priorities[idx] = priority

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        samples = [self.buffer[i] for i in indices]
        
        states = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
        actions = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)
        next_states = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.float32)
        costs = torch.tensor(np.array([s[3] for s in samples]), dtype=torch.float32)
        
        return states, actions, next_states, costs

def finetune(dyn_model, cost_model, buffer, device, s_mean, s_std, c_mean, c_std, batch_size=256):
    data = buffer.sample(batch_size)
    if data is None:
        return 0.0, 0.0

    states, actions, next_states, costs = [d.to(device) for d in data]

    norm_s = (states - s_mean) / s_std
    norm_next_s = (next_states - s_mean) / s_std
    norm_c = (costs - c_mean) / c_std

    opt_d = optim.AdamW(dyn_model.parameters(), lr=1e-5, weight_decay=1e-4)
    opt_c = optim.AdamW(cost_model.parameters(), lr=1e-5, weight_decay=1e-2)

    criterion_d = nn.SmoothL1Loss()
    criterion_c = nn.SmoothL1Loss()

    dyn_model.train()
    cost_model.train()

    opt_d.zero_grad()
    pred_next = dyn_model(norm_s, actions)
    loss_d = criterion_d(pred_next, norm_next_s)
    loss_d.backward()
    opt_d.step()

    opt_c.zero_grad()
    tau = torch.cat([norm_s, actions], dim=-1)
    pred_cost = cost_model(tau)
    loss_c = criterion_c(pred_cost, norm_c.squeeze(-1))
    loss_c.backward()
    opt_c.step()

    return loss_d.item(), loss_c.item()

def main():
    max_planes = 10
    enable_accel = True
    enable_wind = True

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("videos", exist_ok=True)
    os.makedirs("weights/weights", exist_ok=True)

    run = aim.Run(experiment="mpc_finetuning")

    env = AirTrafficEnv(
        render_mode="rgb_array", 
        max_planes=max_planes, 
        enable_acceleration=enable_accel, 
        enable_wind=enable_wind
    )
    
    nx = max_planes * 6 + (2 if enable_wind else 0)
    nu = max_planes * (2 if enable_accel else 1)
    t_horizon = 15

    dynamics_model = LearnedDynamics(nx, nu).to(device)
    cost_model = LearnedCost(nx, nu).to(device)

    dynamics_model.load_state_dict(torch.load("weights/dynamics_model.pth", map_location=device, weights_only=True))
    cost_model.load_state_dict(torch.load("weights/cost_model.pth", map_location=device, weights_only=True))
    
    norms = torch.load("weights/normalization_params.pt", map_location=device, weights_only=True)
    s_mean = norms["state_mean"]
    s_std = norms["state_std"]
    c_mean = norms["cost_mean"]
    c_std = norms["cost_std"]

    wrapped_dyn = WrappedDynamics(dynamics_model, s_mean, s_std).to(device)
    wrapped_cost = WrappedCost(cost_model, nx, s_mean, s_std, c_mean, c_std).to(device)
    
    buffer = PrioritizedBuffer(capacity=10000)

    ctrl = mpc.MPC(
        n_state=nx,
        n_ctrl=nu,
        T=t_horizon,
        u_lower=-1.0,
        u_upper=1.0,
        lqr_iter=20,
        exit_unconverged=False,
        eps=1e-2,
        n_batch=1,
        backprop=False,
        grad_method=mpc.GradMethods.AUTO_DIFF,
        verbose=-1
    )

    for i in range(1, 201):
        print(f"Episode {i}")
        env.reset()
        frames = []
        episode_reward = 0.0
        
        dynamics_model.eval()
        cost_model.eval()

        for step in range(500):
            print(f"Step {step}")
            raw_state = env.get_mpc_state()
            state_tensor = torch.tensor(raw_state, dtype=torch.float32, device=device).unsqueeze(0)

            try:
                nominal_states, nominal_actions, nominal_objs = ctrl(state_tensor, wrapped_cost, wrapped_dyn)
                optimal_action = nominal_actions[0].cpu().detach().numpy()[0]
            except Exception as e:
                print(f"Error occurred: {e}")
                traceback.print_exc()
                optimal_action = np.zeros(nu, dtype=np.float32)

            if np.any(np.isnan(optimal_action)):
                optimal_action = np.zeros_like(optimal_action)

            actions = {}
            for idx, agent in enumerate(env.agents):
                if raw_state[idx * 6 + 4] == 1.0:
                    action_idx = idx * 2 if enable_accel else idx
                    if enable_accel:
                        actions[agent] = np.array([optimal_action[action_idx], optimal_action[action_idx + 1]], dtype=np.float32)
                    else:
                        actions[agent] = np.array([optimal_action[action_idx]], dtype=np.float32)
            # print(f"State: {raw_state}")
            # print(f"Actions: {actions}")

            obs, step_rewards, term, trunc, infos = env.step(actions)
            raw_next_state = env.get_mpc_state()
            step_cost = -sum(step_rewards.values())
            episode_reward += sum(step_rewards.values())

            with torch.no_grad():
                action_tensor = torch.tensor(optimal_action, dtype=torch.float32, device=device).unsqueeze(0)
                next_state_tensor = torch.tensor(raw_next_state, dtype=torch.float32, device=device).unsqueeze(0)
                
                norm_s = (state_tensor - s_mean) / s_std
                norm_next_s_target = (next_state_tensor - s_mean) / s_std
                
                pred_next_s = dynamics_model(norm_s, action_tensor)
                error = nn.functional.mse_loss(pred_next_s, norm_next_s_target).item()

            buffer.add(raw_state, optimal_action, raw_next_state, step_cost, error)

            run.track(sum(step_rewards.values()), name="reward", context={"type": "step"}, step=step + ((i-1) * 500))

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if all(term.values()) or all(trunc.values()):
                break

        run.track(episode_reward, name="reward", context={"type": "episode"}, step=i)
        
        env.save_video("videos", frames, filename=f"mpc_videos_step_{i - 1}.mp4", fps=30)

        total_d_loss = 0.0
        total_c_loss = 0.0
        finetune_steps = 50
        
        for _ in range(finetune_steps):
            d_loss, c_loss = finetune(dynamics_model, cost_model, buffer, device, s_mean, s_std, c_mean, c_std)
            total_d_loss += d_loss
            total_c_loss += c_loss

        run.track(total_d_loss / finetune_steps, name="loss", context={"model": "dynamics", "phase": "finetune"}, step=i)
        run.track(total_c_loss / finetune_steps, name="loss", context={"model": "cost", "phase": "finetune"}, step=i)

        torch.save(dynamics_model.cpu().state_dict(), f"weights/weights/dynamics_model_{i}.pth")
        torch.save(cost_model.cpu().state_dict(), f"weights/cost_model_{i}.pth")
        dynamics_model.to(device)
        cost_model.to(device)

    env.close()

if __name__ == "__main__":
    main()