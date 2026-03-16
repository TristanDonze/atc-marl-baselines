import json
from pathlib import Path

def build_stage_paths(root, stage, seed):
    run_dir = root / stage.name / f"seed_{seed}"
    paths = {
        "run_dir": run_dir,
        "checkpoint_dir": run_dir / "checkpoints",
        "video_dir": run_dir / "videos",
        "config_path": run_dir / "config.json",
        "final_model": run_dir / "checkpoints" / "final_model.pth",
        "final_optimizer": run_dir / "checkpoints" / "final_optimizer.pth",
        "final_vecnormalize": run_dir / "checkpoints" / "final_vec_normalize.pkl",
    }
    for key, value in paths.items():
        if key.endswith("_dir") or key == "checkpoint_dir":
            value.mkdir(parents=True, exist_ok=True)
    return paths

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

def save_stage_config(paths, stage, seed, num_envs, warm_start, ppo_kwargs):
    payload = {
        "stage": {
            "name": stage.name,
            "max_planes": stage.max_planes,
            "spawn_planes": stage.spawn_planes,
            "num_envs": stage.num_envs,
            "enable_acceleration": stage.enable_acceleration,
            "acceleration_scale": stage.acceleration_scale,
            "enable_wind": stage.enable_wind,
            "include_wind_in_obs": stage.include_wind_in_obs,
            "total_timesteps": stage.total_timesteps
        },
        "seed": seed,
        "num_envs": num_envs,
        "ppo": to_jsonable(ppo_kwargs),
        "warm_start": str(warm_start) if warm_start is not None else None,
    }
    with paths["config_path"].open("w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2)

def compatible_warm_start_model(warm_start_model, previous_stage, current_stage, stage_space_signature_fn):
    if warm_start_model is None or previous_stage is None:
        return None
    if not warm_start_model.exists():
        return None

    previous_signature = stage_space_signature_fn(previous_stage)
    current_signature = stage_space_signature_fn(current_stage)
    if previous_signature != current_signature:
        return None

    return warm_start_model

def compatible_warm_start_normalizer(warm_start_normalizer, previous_stage, current_stage, stage_space_signature_fn):
    if warm_start_normalizer is None or previous_stage is None:
        return None
    if not warm_start_normalizer.exists():
        return None

    previous_signature = stage_space_signature_fn(previous_stage)
    current_signature = stage_space_signature_fn(current_stage)
    if previous_signature[0] != current_signature[0]:
        return None
    return warm_start_normalizer
