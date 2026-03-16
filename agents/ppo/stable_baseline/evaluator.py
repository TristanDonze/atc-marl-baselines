import secrets
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

from agents.ppo.stable_baseline.config import StageConfig
from agents.ppo.stable_baseline.env import make_air_traffic_env, make_vector_env
from agents.ppo.stable_baseline.utils import build_stage_paths

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