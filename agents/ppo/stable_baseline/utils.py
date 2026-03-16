import json
from pathlib import Path
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from agents.ppo.stable_baseline.config import StageConfig
from agents.ppo.stable_baseline.env import stage_space_signature

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

def save_stage_config(paths: dict[str, Path], stage: StageConfig, seed: int, num_envs: int, warm_start: Path | None, ppo_kwargs: dict):
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