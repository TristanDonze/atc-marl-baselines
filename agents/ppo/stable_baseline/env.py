import numpy as np
import supersuit as ss
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize
from gym_air_traffic.envs.air_traffic_env import AirTrafficEnv
from agents.ppo.stable_baseline.config import StageConfig

class BooleanDoneWrapper(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs, rewards, np.asarray(dones, dtype=np.bool_), infos

def make_air_traffic_env(stage: StageConfig, render_mode=None) -> AirTrafficEnv:
    return AirTrafficEnv(
        render_mode=render_mode,
        max_planes=stage.max_planes,
        spawn_planes=stage.spawn_planes,
        enable_acceleration=stage.enable_acceleration,
        acceleration_scale=stage.acceleration_scale,
        enable_wind=stage.enable_wind,
        include_wind_in_obs=stage.include_wind_in_obs,
    )

def stage_space_signature(stage: StageConfig) -> tuple[int, int, int, int]:
    env = make_air_traffic_env(stage, render_mode=None)
    try:
        return env.obs_dim, env.action_dim, env.max_planes, env.neighbor_feature_dim
    finally:
        env.close()

def build_observation_layout(stage: StageConfig) -> dict[str, int]:
    env = make_air_traffic_env(stage, render_mode=None)
    try:
        return {
            "self_feature_dim": env.obs_dim - ((env.max_planes - 1) * env.neighbor_feature_dim),
            "neighbor_feature_dim": env.neighbor_feature_dim,
            "max_neighbors": env.max_planes - 1,
        }
    finally:
        env.close()

def make_vector_env(
    stage: StageConfig,
    seed: int,
    num_envs: int,
    monitor_dir,
    normalize_path=None,
    training: bool = True,
):
    base_env = make_air_traffic_env(stage, render_mode=None)
    base_env.reset(seed=seed)
    env = ss.pettingzoo_env_to_vec_env_v1(base_env)
    env = ss.concat_vec_envs_v1(
        env,
        num_envs,
        num_cpus=1,
        base_class="stable_baselines3",
    )
    env = BooleanDoneWrapper(env)
    env = VecMonitor(env, str(monitor_dir)) if monitor_dir is not None else VecMonitor(env)

    if normalize_path is not None and normalize_path.exists():
        env = VecNormalize.load(str(normalize_path), env)
        env.training = training
        env.norm_reward = training
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=training, clip_obs=10.0, clip_reward=10.0)

    return env