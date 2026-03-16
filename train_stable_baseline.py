import argparse
from pathlib import Path
import torch as th
from agents.ppo.stable_baseline.config import StageConfig
from agents.ppo.stable_baseline.trainer import train_curriculum

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGE_BY_NAME), default=[stage.name for stage in CURRICULUM])
    return parser.parse_args()

def main():
    args = parse_args()
    selected_stages = [STAGE_BY_NAME[name] for name in args.stages]
    train_curriculum(
        stages=selected_stages,
        seeds=args.seeds,
        artifact_root=args.artifact_root,
        num_envs_override=args.num_envs,
        ppo_kwargs=PPO_KWARGS,
        policy_kwargs=POLICY_KWARGS,
        feature_extractor_kwargs=FEATURE_EXTRACTOR_KWARGS,
    )

if __name__ == "__main__":
    main()