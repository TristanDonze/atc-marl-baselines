import torch
import numpy as np
import logging
from agents.ppo.implementation.network import AirTrafficActorCriticNetwork
from agents.ppo.implementation.buffer import RolloutBuffer

LR = 1.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CRITIC_LOSS_WEIGHT = 0.5
ENTROPY_TERM_WEIGHT = 0.005
PPO_EPOCHS = 10
PPO_MINIBATCH_SIZE = 512
PPO_CLIP_EPS = 0.2
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.03

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ProximalPolicyOptimizationAgent")

class ProximalPolicyOptimizationAgent:
    def __init__(self, env_vec, num_updates, rollout_steps, run_aim, stage_name, layout):
        self.env = env_vec
        self.num_envs = getattr(self.env, 'num_envs', 1)
        self.num_updates = num_updates
        self.rollout_steps = rollout_steps
        self.run = run_aim
        self.stage_name = stage_name
        self.train_done = 0
        self.total_env_steps = 0

        if torch.cuda.is_available(): 
            self.device = "cuda"
        else: 
            self.device = "cpu"

        self.current_state, _ = self.env.reset()
        
        if hasattr(self.env.action_space, 'shape') and len(self.env.action_space.shape) > 0:
            self.action_size = self.env.action_space.shape[0]
            self.action_type = "continuous"
        else:
            self.action_size = self.env.action_space.n
            self.action_type = "discrete"

        self.ac_nn = AirTrafficActorCriticNetwork(
            self_feature_dim=layout["self_feature_dim"],
            neighbor_feature_dim=layout["neighbor_feature_dim"],
            max_neighbors=layout["max_neighbors"],
            n_actions=self.action_size,
            action_space_type=self.action_type
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.ac_nn.parameters(), lr=LR, eps=1e-5)
        self.rollout_buffer = RolloutBuffer(self.rollout_steps, self.num_envs, self.device)

        self.current_ep_rewards = np.zeros(self.num_envs)
        self.completed_ep_rewards = []

    @property
    def model(self):
        return self.ac_nn

    @torch.no_grad()
    def sample_action(self, state):
        mean, log_std = self.ac_nn.get_action_distribution_params(state)
        value = self.ac_nn.get_state_value(state).squeeze(-1)
        
        if self.action_type == "continuous":
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=mean)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.cpu().numpy(), log_prob, value

    def evaluate_actions(self, states, actions):
        mean, log_std = self.ac_nn.get_action_distribution_params(states)
        values = self.ac_nn.get_state_value(states).squeeze(-1)
        
        if self.action_type == "continuous":
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=mean)
            log_probs = dist.log_prob(actions.squeeze(-1))
            entropy = dist.entropy()
            
        return log_probs, entropy, values

    def _collect_rollout(self):
        rollout_completed_agents = 0
        rollout_collision_agents = 0
        rollout_gate_pass_agents = 0

        for _ in range(self.rollout_steps):
            state_t = torch.tensor(self.current_state, dtype=torch.float32).to(self.device)
            action, log_prob, value = self.sample_action(state_t)
            
            obs, rewards, dones, infos = self.env.step(action)
            
            self.rollout_buffer.add(self.current_state, action, rewards, dones, value, log_prob)
            self.current_state = obs
            self.total_env_steps += self.num_envs

            for i, done in enumerate(dones):
                self.current_ep_rewards[i] += float(rewards[i])
                
                if done:
                    self.completed_ep_rewards.append(self.current_ep_rewards[i])
                    self.current_ep_rewards[i] = 0.0
                    
                    rollout_completed_agents += 1
                    if isinstance(infos, tuple) and len(infos) > i and isinstance(infos[i], dict):
                        if infos[i].get("termination_reason") == "collision":
                            rollout_collision_agents += 1
                        if infos[i].get("gate_passed", False):
                            rollout_gate_pass_agents += 1
                    elif isinstance(infos, dict):
                        if "termination_reason" in infos and isinstance(infos["termination_reason"], np.ndarray):
                            if infos["termination_reason"][i] == "collision":
                                rollout_collision_agents += 1
                        if "gate_passed" in infos and isinstance(infos["gate_passed"], np.ndarray):
                            if infos["gate_passed"][i]:
                                rollout_gate_pass_agents += 1

        if rollout_completed_agents > 0:
            self.run.track(rollout_collision_agents / rollout_completed_agents, name="traffic/collision_rate_rollout", step=self.total_env_steps)
            self.run.track(rollout_gate_pass_agents / rollout_completed_agents, name="traffic/gate_pass_rate_rollout", step=self.total_env_steps)
        self.run.track(rollout_completed_agents, name="traffic/completed_agents_rollout", step=self.total_env_steps)

        if len(self.completed_ep_rewards) > 0:
            avg_reward = float(np.mean(self.completed_ep_rewards))
            self.run.track(avg_reward, name="reward/train_avg_per_episode", step=self.total_env_steps)
            self.completed_ep_rewards.clear()

    @torch.no_grad()
    def compute_gae(self, rewards, dones, values, next_value, next_done):
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            delta = rewards[t] + GAMMA * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
        returns = advantages + values
        return returns, advantages

    def _iterate_minibatches(self, n, batch_size):
        idx = torch.randperm(n, device=self.device)
        for start in range(0, n, batch_size):
            yield idx[start:start + batch_size]

    def train(self):
        while self.train_done < self.num_updates:
            self._collect_rollout()
            states, actions, rewards, dones, values, logp_old = self.rollout_buffer.get_tensors()

            with torch.no_grad():
                next_state_t = torch.tensor(self.current_state, dtype=torch.float32).to(self.device)
                next_value = self.ac_nn.get_state_value(next_state_t).squeeze(-1)
                next_done = torch.tensor(np.zeros(self.num_envs), dtype=torch.float32).to(self.device)

            returns, advantages = self.compute_gae(rewards, dones, values, next_value, next_done)
            
            b_states = states.view(-1, states.shape[-1])
            if self.action_type == "continuous":
                b_actions = actions.view(-1, actions.shape[-1])
            else:
                b_actions = actions.view(-1)
            b_advantages = advantages.view(-1)
            b_returns = returns.view(-1)
            b_logp_old = logp_old.view(-1)
            b_values = values.view(-1)

            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            n_samples = b_states.shape[0]
            avg_loss_actor, avg_loss_critic, avg_entropy, avg_kl, avg_clipfrac = 0.0, 0.0, 0.0, 0.0, 0.0
            num_minibatches_processed = 0
            early_stop = False

            for epoch in range(PPO_EPOCHS):
                for mb_idx in self._iterate_minibatches(n_samples, PPO_MINIBATCH_SIZE):
                    mb_states = b_states[mb_idx]
                    mb_actions = b_actions[mb_idx]
                    mb_adv = b_advantages[mb_idx]
                    mb_ret = b_returns[mb_idx]
                    mb_logp_old = b_logp_old[mb_idx]
                    mb_v_old = b_values[mb_idx]

                    logp_new, entropy, v_pred = self.evaluate_actions(mb_states, mb_actions)

                    prob_ratio = torch.exp(logp_new - mb_logp_old)
                    surrogate1 = prob_ratio * mb_adv
                    surrogate2 = torch.clamp(prob_ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * mb_adv
                    loss_actor = -torch.mean(torch.min(surrogate1, surrogate2))

                    v_pred_clipped = mb_v_old + torch.clamp(v_pred - mb_v_old, -PPO_CLIP_EPS, PPO_CLIP_EPS)
                    value_loss_unclipped = (v_pred - mb_ret).pow(2)
                    value_loss_clipped = (v_pred_clipped - mb_ret).pow(2)
                    loss_critic = 0.5 * torch.mean(torch.max(value_loss_unclipped, value_loss_clipped))

                    ent = entropy.mean()
                    loss = loss_actor + CRITIC_LOSS_WEIGHT * loss_critic - ENTROPY_TERM_WEIGHT * ent

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ac_nn.parameters(), max_norm=MAX_GRAD_NORM)
                    self.optimizer.step()

                    with torch.no_grad():
                        kl = torch.mean(mb_logp_old - logp_new)
                        clipfrac = torch.mean((torch.abs(prob_ratio - 1.0) > PPO_CLIP_EPS).float())

                    avg_loss_actor += loss_actor.item()
                    avg_loss_critic += loss_critic.item()
                    avg_entropy += ent.item()
                    avg_kl += kl.item()
                    avg_clipfrac += clipfrac.item()
                    num_minibatches_processed += 1

                    if TARGET_KL and kl.item() > TARGET_KL: 
                        early_stop = True
                        break
                if early_stop: 
                    break

            if num_minibatches_processed > 0:
                self.run.track(avg_loss_actor / num_minibatches_processed, name="loss/actor_clip", step=self.total_env_steps)
                self.run.track(avg_loss_critic / num_minibatches_processed, name="loss/value_clip", step=self.total_env_steps)
                self.run.track(avg_entropy / num_minibatches_processed, name="entropy", step=self.total_env_steps)
                self.run.track(avg_kl / num_minibatches_processed, name="kl", step=self.total_env_steps)
                self.run.track(avg_clipfrac / num_minibatches_processed, name="clipfrac", step=self.total_env_steps)
            
            self.rollout_buffer.clear()
            self.train_done += 1