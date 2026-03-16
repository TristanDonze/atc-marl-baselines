import argparse
import math
from pathlib import Path
import random
import numpy as np
import torch
import aim
from agents.ppo.stable_baseline.config import StageConfig
from agents.ppo.implementation.env import make_vector_env, build_observation_layout, stage_space_signature
from agents.ppo.implementation.utils import build_stage_paths, save_stage_config, compatible_warm_start_model, compatible_warm_start_normalizer
from agents.ppo.implementation.agent import ProximalPolicyOptimizationAgent

EXPERIMENT_ROOT = Path("experiments/ppo_implementation")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_NUM_ENVS = None

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

    StageConfig(
        name="stage7_ten_planes_acceleration_wind",
        max_planes=10,
        spawn_planes=10,
        num_envs=12,
        enable_acceleration=True,
        acceleration_scale=1.0,
        enable_wind=True,
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

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_stage(stage, seed, artifact_root, num_envs, warm_start_model, warm_start_optimizer, warm_start_normalizer, previous_stage, run_aim):
    paths = build_stage_paths(artifact_root, stage, seed)
    save_stage_config(paths, stage, seed, num_envs, warm_start_model, {})

    warm_start_model = compatible_warm_start_model(warm_start_model, previous_stage, stage, stage_space_signature)
    warm_start_normalizer = compatible_warm_start_normalizer(warm_start_normalizer, previous_stage, stage, stage_space_signature)
    if warm_start_model is None or warm_start_optimizer is None or not warm_start_optimizer.exists():
        warm_start_optimizer = None

    env = make_vector_env(
        stage=stage,
        seed=seed,
        num_envs=num_envs,
        normalize_path=warm_start_normalizer,
        training=True,
    )

    layout = build_observation_layout(stage)
    
    rollout_steps = 1024
    num_updates = math.ceil(stage.total_timesteps / (env.num_envs * rollout_steps))

    agent = ProximalPolicyOptimizationAgent(
        env_vec=env,
        num_updates=num_updates,
        rollout_steps=rollout_steps,
        run_aim=run_aim,
        stage_name=stage.name,
        layout=layout
    )

    if warm_start_model is not None and warm_start_model.exists():
        agent.ac_nn.load_state_dict(torch.load(warm_start_model, map_location=agent.device))
    if warm_start_optimizer is not None:
        optimizer_state = torch.load(warm_start_optimizer, map_location=agent.device)
        agent.optimizer.load_state_dict(optimizer_state)
        for state in agent.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(agent.device)

    agent.train()

    torch.save(agent.ac_nn.state_dict(), paths["final_model"])
    torch.save(agent.optimizer.state_dict(), paths["final_optimizer"])
    env.save(paths["final_vecnormalize"])
    env.close()

    return paths["final_model"], paths["final_optimizer"], paths["final_vecnormalize"]

def train_curriculum(stages, seeds, artifact_root, num_envs_override):
    for seed in seeds:
        set_random_seed(seed)
        warm_start_model = None
        warm_start_optimizer = None
        warm_start_normalizer = None
        previous_stage = None
        
        run_aim = aim.Run(experiment=f"ppo_curriculum_seed_{seed}")
        
        for stage in stages:
            stage_num_envs = num_envs_override if num_envs_override is not None else stage.num_envs
            warm_start_model, warm_start_optimizer, warm_start_normalizer = train_stage(
                stage=stage,
                seed=seed,
                artifact_root=artifact_root,
                num_envs=stage_num_envs,
                warm_start_model=warm_start_model,
                warm_start_optimizer=warm_start_optimizer,
                warm_start_normalizer=warm_start_normalizer,
                previous_stage=previous_stage,
                run_aim=run_aim
            )
            previous_stage = stage
        
        run_aim.close()

def main():
    args = parse_args()
    selected_stages = [STAGE_BY_NAME[name] for name in args.stages]
    train_curriculum(
        stages=selected_stages,
        seeds=args.seeds,
        artifact_root=args.artifact_root,
        num_envs_override=args.num_envs,
    )

if __name__ == "__main__":
    main()
