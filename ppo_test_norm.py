import argparse
import json
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
import torch as th

import matplotlib.pyplot as plt
import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize

from agents.ppo import StructuredAirTrafficExtractor
from gym_air_traffic.envs.air_traffic_env import AirTrafficEnv


EXPERIMENT_ROOT = Path("experiments/ppo_ten_plane_curriculum")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_NUM_ENVS = None
PPO_KWARGS = {
    "learning_rate": 1.5e-4,
    "n_steps": 1024,
    "batch_size": 512,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.005,
    "clip_range": 0.2,
    "target_kl": 0.03,
}
POLICY_KWARGS = {
    "activation_fn": th.nn.ReLU,
    "net_arch": dict(
        pi=[256, 256],
        vf=[512, 512, 256],
    ),
}
FEATURE_EXTRACTOR_KWARGS = {
    "features_dim": 384,
    "self_hidden_dim": 192,
    "neighbor_hidden_dim": 192,
}


@dataclass(frozen=True)
class StageConfig:
    name: str
    max_planes: int
    spawn_planes: int
    num_envs: int
    enable_acceleration: bool
    acceleration_scale: float
    enable_wind: bool
    include_wind_in_obs: bool
    total_timesteps: int


CURRICULUM = (
    StageConfig(
        name="stage1_two_planes_fixed_speed",
        max_planes=10,
        spawn_planes=2,
        num_envs=8,
        enable_acceleration=True,
        acceleration_scale=0.0,
        enable_wind=False,
        include_wind_in_obs=False,
        total_timesteps=2_000_000,
    ),

    StageConfig(
        name="stage2_four_planes_fixed_speed",
        max_planes=10,
        spawn_planes=4,
        num_envs=8,
        enable_acceleration=True,
        acceleration_scale=0.0,
        enable_wind=False,
        include_wind_in_obs=False,
        total_timesteps=3_000_000,
    ),

    StageConfig(
        name="stage3_six_planes_fixed_speed",
        max_planes=10,
        spawn_planes=6,
        num_envs=8,
        enable_acceleration=True,
        acceleration_scale=0.0,
        enable_wind=False,
        include_wind_in_obs=False,
        total_timesteps=4_000_000,
    ),

    StageConfig(
        name="stage4_eight_planes_fixed_speed",
        max_planes=10,
        spawn_planes=8,
        num_envs=12,
        enable_acceleration=True,
        acceleration_scale=0.0,
        enable_wind=False,
        include_wind_in_obs=False,
        total_timesteps=4_000_000,
    ),

    StageConfig(
        name="stage5_ten_planes_fixed_speed",
        max_planes=10,
        spawn_planes=10,
        num_envs=12,
        enable_acceleration=True,
        acceleration_scale=0.0,
        enable_wind=False,
        include_wind_in_obs=False,
        total_timesteps=6_000_000,
    ),

    StageConfig(
        name="stage6_ten_planes_acceleration",
        max_planes=10,
        spawn_planes=10,
        num_envs=12,
        enable_acceleration=True,
        acceleration_scale=1.0,
        enable_wind=False,
        include_wind_in_obs=False,
        total_timesteps=6_000_000,
    ),

)

STAGE_BY_NAME = {stage.name: stage for stage in CURRICULUM}


class BooleanDoneWrapper(VecEnvWrapper):
    """VecNormalize expects boolean done arrays."""

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs, rewards, np.asarray(dones, dtype=np.bool_), infos


class BestModelCallback(BaseCallback):
    def __init__(self, checkpoint_dir: Path, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.checkpoint_dir = checkpoint_dir
        self.best_mean_reward = -np.inf
        self.best_model_path = checkpoint_dir / "best_model"
        self.best_vecnormalize_path = checkpoint_dir / "best_vec_normalize.pkl"
        self.total_completed_agents = 0
        self.total_collision_agents = 0
        self.total_gate_pass_agents = 0
        self.rollout_completed_agents = 0
        self.rollout_collision_agents = 0
        self.rollout_gate_pass_agents = 0

    def _record_completed_agent_metrics(self) -> None:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None or dones is None:
            return

        for done, info in zip(dones, infos):
            if not done:
                continue

            self.total_completed_agents += 1
            self.rollout_completed_agents += 1

            if info.get("termination_reason") == "collision":
                self.total_collision_agents += 1
                self.rollout_collision_agents += 1

            if info.get("gate_passed", False):
                self.total_gate_pass_agents += 1
                self.rollout_gate_pass_agents += 1

    def _on_step(self) -> bool:
        self._record_completed_agent_metrics()

        if len(self.model.ep_info_buffer) == 0:
            return True

        mean_reward = float(np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
        if mean_reward <= self.best_mean_reward:
            return True

        self.best_mean_reward = mean_reward
        self.model.save(str(self.best_model_path))

        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            vec_normalize.save(str(self.best_vecnormalize_path))

        if self.verbose > 0:
            print(f"Saved new best checkpoint with mean reward {self.best_mean_reward:.2f}")
        return True

    def _on_rollout_end(self) -> None:
        if self.rollout_completed_agents > 0:
            self.logger.record(
                "traffic/collision_rate_rollout",
                self.rollout_collision_agents / self.rollout_completed_agents,
            )
            self.logger.record(
                "traffic/gate_pass_rate_rollout",
                self.rollout_gate_pass_agents / self.rollout_completed_agents,
            )

        if self.total_completed_agents > 0:
            self.logger.record(
                "traffic/collision_rate_total",
                self.total_collision_agents / self.total_completed_agents,
            )
            self.logger.record(
                "traffic/gate_pass_rate_total",
                self.total_gate_pass_agents / self.total_completed_agents,
            )

        self.logger.record("traffic/completed_agents_rollout", self.rollout_completed_agents)
        self.logger.record("traffic/completed_agents_total", self.total_completed_agents)

        self.rollout_completed_agents = 0
        self.rollout_collision_agents = 0
        self.rollout_gate_pass_agents = 0


def build_stage_paths(root: Path, stage: StageConfig, seed: int) -> dict[str, Path]:
    run_dir = root / stage.name / f"seed_{seed}"
    paths = {
        "run_dir": run_dir,
        "monitor_dir": run_dir / "monitor",
        "tensorboard_dir": run_dir / "tensorboard",
        "checkpoint_dir": run_dir / "checkpoints",
        "video_dir": run_dir / "videos",
        "plot_path": run_dir / "reward_plot.png",
        "config_path": run_dir / "config.json",
        "final_model": run_dir / "checkpoints" / "final_model",
        "final_vecnormalize": run_dir / "checkpoints" / "final_vec_normalize.pkl",
    }
    for key, value in paths.items():
        if key.endswith("_dir") or key == "checkpoint_dir":
            value.mkdir(parents=True, exist_ok=True)
    return paths


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


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    return value


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


def build_policy_kwargs(stage: StageConfig) -> dict:
    return {
        **POLICY_KWARGS,
        "features_extractor_class": StructuredAirTrafficExtractor,
        "features_extractor_kwargs": {
            **build_observation_layout(stage),
            **FEATURE_EXTRACTOR_KWARGS,
        },
    }


def build_ppo_kwargs(stage: StageConfig) -> dict:
    return {
        **PPO_KWARGS,
        "policy_kwargs": build_policy_kwargs(stage),
    }


def make_vector_env(
    stage: StageConfig,
    seed: int,
    num_envs: int,
    monitor_dir: Path | None,
    normalize_path: Path | None = None,
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


def save_stage_config(paths: dict[str, Path], stage: StageConfig, seed: int, num_envs: int, warm_start: Path | None):
    payload = {
        "stage": asdict(stage),
        "seed": seed,
        "num_envs": num_envs,
        "ppo": to_jsonable(build_ppo_kwargs(stage)),
        "warm_start": str(warm_start) if warm_start is not None else None,
    }
    with paths["config_path"].open("w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2)


def create_model(env, stage: StageConfig, tensorboard_dir: Path) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(tensorboard_dir),
        **build_ppo_kwargs(stage),
    )


def compatible_warm_start_model(
    warm_start_model: Path | None,
    previous_stage: StageConfig | None,
    current_stage: StageConfig,
) -> Path | None:
    if warm_start_model is None or previous_stage is None:
        return None
    if not warm_start_model.with_suffix(".zip").exists():
        return None

    previous_signature = stage_space_signature(previous_stage)
    current_signature = stage_space_signature(current_stage)
    if previous_signature != current_signature:
        print(
            "Skipping warm start because stage spaces changed: "
            f"{previous_stage.name} {previous_signature} -> {current_stage.name} {current_signature}"
        )
        return None

    return warm_start_model


def compatible_warm_start_normalizer(
    warm_start_normalizer: Path | None,
    previous_stage: StageConfig | None,
    current_stage: StageConfig,
) -> Path | None:
    if warm_start_normalizer is None or previous_stage is None:
        return None
    if not warm_start_normalizer.exists():
        return None

    previous_signature = stage_space_signature(previous_stage)
    current_signature = stage_space_signature(current_stage)
    if previous_signature[0] != current_signature[0]:
        print(
            "Skipping VecNormalize warm start because observation dimensions changed: "
            f"{previous_stage.name} obs={previous_signature[0]} -> {current_stage.name} obs={current_signature[0]}"
        )
        return None
    return warm_start_normalizer


def load_or_create_model(
    env,
    stage: StageConfig,
    tensorboard_dir: Path,
    warm_start_model: Path | None,
) -> tuple[PPO, bool]:
    if warm_start_model is None or not warm_start_model.with_suffix(".zip").exists():
        return create_model(env, stage, tensorboard_dir), True

    model = PPO.load(str(warm_start_model), env=env, device="auto")
    model.tensorboard_log = str(tensorboard_dir)
    model.verbose = 1
    return model, False


def plot_training_curve(paths: dict[str, Path], timesteps: int, title: str):
    try:
        plt.figure(figsize=(14, 8))
        results_plotter.plot_results(
            [str(paths["monitor_dir"])],
            timesteps,
            results_plotter.X_TIMESTEPS,
            title,
        )
        plt.tight_layout()
        plt.savefig(paths["plot_path"])
    except (IndexError, ValueError) as exc:
        print(f"Skipping reward plot for {title}: {exc}")
    finally:
        plt.close("all")


def train_stage(
    stage: StageConfig,
    seed: int,
    artifact_root: Path,
    num_envs: int,
    warm_start_model: Path | None,
    warm_start_normalizer: Path | None,
    previous_stage: StageConfig | None,
) -> tuple[Path, Path]:
    paths = build_stage_paths(artifact_root, stage, seed)
    save_stage_config(paths, stage, seed, num_envs, warm_start_model)

    warm_start_model = compatible_warm_start_model(warm_start_model, previous_stage, stage)
    warm_start_normalizer = compatible_warm_start_normalizer(warm_start_normalizer, previous_stage, stage)

    env = make_vector_env(
        stage=stage,
        seed=seed,
        num_envs=num_envs,
        monitor_dir=paths["monitor_dir"],
        normalize_path=warm_start_normalizer,
        training=True,
    )
    model, is_fresh_model = load_or_create_model(env, stage, paths["tensorboard_dir"], warm_start_model)
    callback = BestModelCallback(paths["checkpoint_dir"])

    print(f"Training {stage.name} | seed={seed} | num_envs={num_envs}")
    model.learn(
        total_timesteps=stage.total_timesteps,
        tb_log_name=f"{stage.name}_seed_{seed}",
        callback=callback,
        reset_num_timesteps=is_fresh_model,
    )

    model.save(str(paths["final_model"]))
    vec_normalize = model.get_vec_normalize_env()
    if vec_normalize is not None:
        vec_normalize.save(str(paths["final_vecnormalize"]))

    plot_training_curve(paths, model.num_timesteps, f"{stage.name} Seed {seed}")
    env.close()
    return paths["final_model"], paths["final_vecnormalize"]


def resolve_model_seed(stage: StageConfig, artifact_root: Path, model_seed: int | None) -> int:
    if model_seed is not None:
        return model_seed

    stage_dir = artifact_root / stage.name
    candidate_dirs = []
    for run_dir in stage_dir.glob("seed_*"):
        if not run_dir.is_dir():
            continue
        try:
            seed_value = int(run_dir.name.split("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        checkpoint = run_dir / "checkpoints" / "final_model.zip"
        if checkpoint.exists():
            candidate_dirs.append((seed_value, run_dir))

    if not candidate_dirs:
        raise FileNotFoundError(f"No trained checkpoints found under {stage_dir}")

    candidate_dirs.sort(key=lambda item: item[0])
    return candidate_dirs[-1][0]


def evaluate_stage(stage: StageConfig, model_seed: int | None, artifact_root: Path, episode_seed: int | None):
    model_seed = resolve_model_seed(stage, artifact_root, model_seed)
    paths = build_stage_paths(artifact_root, stage, model_seed)
    model_path = paths["final_model"]
    normalizer_path = paths["final_vecnormalize"]
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Missing trained model at {model_path.with_suffix('.zip')}")
    if not normalizer_path.exists():
        raise FileNotFoundError(f"Missing VecNormalize stats at {normalizer_path}")

    vec_normalize = make_vector_env(
        stage=stage,
        seed=model_seed,
        num_envs=1,
        monitor_dir=None,
        normalize_path=normalizer_path,
        training=False,
    )
    vec_normalize.training = False
    vec_normalize.norm_reward = False

    model = PPO.load(str(model_path), device="auto")
    test_env = make_air_traffic_env(stage, render_mode="rgb_array")
    effective_episode_seed = episode_seed if episode_seed is not None else secrets.randbelow(2**31 - 1)
    observations, infos = test_env.reset(seed=effective_episode_seed)
    video_name = "latest_eval.mp4" if episode_seed is None else f"eval_seed_{episode_seed}.mp4"
    frames = []

    print(
        f"Evaluating {stage.name} | model_seed={model_seed} | "
        f"episode_seed={effective_episode_seed}"
    )
    try:
        while test_env.steps < 1000:
            actions = {}
            for agent in test_env.agents:
                obs = observations[agent]
                if obs[0] == -1.0:
                    actions[agent] = np.zeros(test_env.action_dim, dtype=np.float32)
                    continue

                normalized_obs = vec_normalize.normalize_obs(np.expand_dims(obs, axis=0))
                action, _ = model.predict(normalized_obs, deterministic=True)
                actions[agent] = action[0]

            observations, rewards, terminations, truncations, infos = test_env.step(actions)
            frame = test_env.render()
            if frame is not None:
                frames.append(frame)

            if all(terminations.values()) or all(truncations.values()):
                print(f"Evaluation finished at step {test_env.steps}")
                break
    finally:
        test_env.save_video(
            paths["video_dir"],
            frames,
            filename=video_name,
            fps=30,
        )
        test_env.close()
        vec_normalize.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate the air-traffic PPO curriculum.")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--artifact-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGE_BY_NAME), default=[stage.name for stage in CURRICULUM])
    parser.add_argument("--eval-stage", choices=tuple(STAGE_BY_NAME), default=CURRICULUM[-1].name)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def train_curriculum(args):
    selected_stages = [STAGE_BY_NAME[name] for name in args.stages]
    for seed in args.seeds:
        set_random_seed(seed)
        warm_start_model = None
        warm_start_normalizer = None
        previous_stage = None
        for stage in selected_stages:
            stage_num_envs = args.num_envs if args.num_envs is not None else stage.num_envs
            warm_start_model, warm_start_normalizer = train_stage(
                stage=stage,
                seed=seed,
                artifact_root=args.artifact_root,
                num_envs=stage_num_envs,
                warm_start_model=warm_start_model,
                warm_start_normalizer=warm_start_normalizer,
                previous_stage=previous_stage,
            )
            previous_stage = stage


def main():
    args = parse_args()
    if args.mode == "train":
        train_curriculum(args)
        return

    evaluate_stage(STAGE_BY_NAME[args.eval_stage], args.model_seed, args.artifact_root, args.seed)


if __name__ == "__main__":
    main()
