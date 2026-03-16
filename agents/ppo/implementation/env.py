import pickle
import numpy as np
import supersuit as ss
from gym_air_traffic.envs.air_traffic_env import AirTrafficEnv

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class VecNormalizeWrapper:
    def __init__(self, venv, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-8):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.last_raw_rewards = np.zeros(self.num_envs, dtype=np.float32)

    def reset(self, seed=None):
        obs, infos = self.venv.reset(seed=seed)
        self.returns = np.zeros(self.num_envs)
        self.last_raw_rewards = np.zeros(self.num_envs, dtype=np.float32)
        return self.normalize_obs(obs), infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.venv.step(actions)
        dones = np.logical_or(terminations, truncations)
        self.last_raw_rewards = np.asarray(rewards, dtype=np.float32)
        self.returns = self.returns * self.gamma + rewards
        normalized_rewards = self.normalize_reward(rewards)
        self.returns[dones] = 0.0
        return self.normalize_obs(obs), normalized_rewards, np.asarray(dones, dtype=np.bool_), infos

    def normalize_obs(self, obs):
        if not self.norm_obs:
            return obs
        if self.training:
            self.obs_rms.update(obs)
        obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
        return np.asarray(obs, dtype=np.float32)

    def normalize_reward(self, rewards):
        if not self.norm_reward:
            return rewards
        if self.training:
            self.ret_rms.update(self.returns)
        rewards = np.clip(rewards / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        return np.asarray(rewards, dtype=np.float32)

    def save(self, path):
        with open(path, "wb") as file_handler:
            pickle.dump({"obs_rms": self.obs_rms, "ret_rms": self.ret_rms}, file_handler)

    def load(self, path):
        with open(path, "rb") as file_handler:
            data = pickle.load(file_handler)
        self.obs_rms = data["obs_rms"]
        self.ret_rms = data["ret_rms"]

    def close(self):
        self.venv.close()

def make_air_traffic_env(stage, render_mode=None):
    return AirTrafficEnv(
        render_mode=render_mode,
        max_planes=stage.max_planes,
        spawn_planes=stage.spawn_planes,
        enable_acceleration=stage.enable_acceleration,
        acceleration_scale=stage.acceleration_scale,
        enable_wind=stage.enable_wind,
        include_wind_in_obs=stage.include_wind_in_obs,
    )

def stage_space_signature(stage):
    env = make_air_traffic_env(stage, render_mode=None)
    try:
        return env.obs_dim, env.action_dim, env.max_planes, env.neighbor_feature_dim
    finally:
        env.close()

def build_observation_layout(stage):
    env = make_air_traffic_env(stage, render_mode=None)
    try:
        return {
            "self_feature_dim": env.obs_dim - ((env.max_planes - 1) * env.neighbor_feature_dim),
            "neighbor_feature_dim": env.neighbor_feature_dim,
            "max_neighbors": env.max_planes - 1,
        }
    finally:
        env.close()

def make_vector_env(stage, seed, num_envs, normalize_path=None, training=True):
    base_env = make_air_traffic_env(stage, render_mode=None)
    base_env.reset(seed=seed)
    env = ss.pettingzoo_env_to_vec_env_v1(base_env)
    env = ss.concat_vec_envs_v1(
        env,
        num_envs,
        num_cpus=1,
        base_class="gymnasium",
    )
    env = VecNormalizeWrapper(env, training=training, norm_obs=True, norm_reward=training, clip_obs=10.0, clip_reward=10.0)
    
    if normalize_path is not None and normalize_path.exists():
        env.load(normalize_path)
        
    return env
