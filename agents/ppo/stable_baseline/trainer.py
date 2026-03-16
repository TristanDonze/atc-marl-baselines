from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from agents.ppo.feature_extractor import StructuredAirTrafficExtractor
from agents.ppo.stable_baseline.config import StageConfig
from agents.ppo.stable_baseline.env import make_vector_env, build_observation_layout
from agents.ppo.stable_baseline.callbacks import BestModelCallback
from agents.ppo.stable_baseline.utils import build_stage_paths, save_stage_config, compatible_warm_start_model, compatible_warm_start_normalizer, plot_training_curve

def build_policy_kwargs(stage: StageConfig, policy_kwargs: dict, feature_extractor_kwargs: dict) -> dict:
    return {
        **policy_kwargs,
        "features_extractor_class": StructuredAirTrafficExtractor,
        "features_extractor_kwargs": {
            **build_observation_layout(stage),
            **feature_extractor_kwargs,
        },
    }

def build_ppo_kwargs(stage: StageConfig, ppo_kwargs: dict, policy_kwargs: dict, feature_extractor_kwargs: dict) -> dict:
    return {
        **ppo_kwargs,
        "policy_kwargs": build_policy_kwargs(stage, policy_kwargs, feature_extractor_kwargs),
    }

def create_model(env, stage: StageConfig, tensorboard_dir: Path, ppo_kwargs: dict) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(tensorboard_dir),
        **ppo_kwargs,
    )

def load_or_create_model(
    env,
    stage: StageConfig,
    tensorboard_dir: Path,
    warm_start_model: Path | None,
    ppo_kwargs: dict,
) -> tuple[PPO, bool]:
    if warm_start_model is None or not warm_start_model.with_suffix(".zip").exists():
        return create_model(env, stage, tensorboard_dir, ppo_kwargs), True

    model = PPO.load(str(warm_start_model), env=env, device="auto")
    model.tensorboard_log = str(tensorboard_dir)
    model.verbose = 1
    return model, False

def train_stage(
    stage: StageConfig,
    seed: int,
    artifact_root: Path,
    num_envs: int,
    warm_start_model: Path | None,
    warm_start_normalizer: Path | None,
    previous_stage: StageConfig | None,
    ppo_kwargs: dict,
    policy_kwargs: dict,
    feature_extractor_kwargs: dict,
) -> tuple[Path, Path]:
    paths = build_stage_paths(artifact_root, stage, seed)
    compiled_ppo_kwargs = build_ppo_kwargs(stage, ppo_kwargs, policy_kwargs, feature_extractor_kwargs)
    save_stage_config(paths, stage, seed, num_envs, warm_start_model, compiled_ppo_kwargs)

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
    model, is_fresh_model = load_or_create_model(env, stage, paths["tensorboard_dir"], warm_start_model, compiled_ppo_kwargs)
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

def train_curriculum(
    stages: list[StageConfig],
    seeds: list[int],
    artifact_root: Path,
    num_envs_override: int | None,
    ppo_kwargs: dict,
    policy_kwargs: dict,
    feature_extractor_kwargs: dict,
):
    for seed in seeds:
        set_random_seed(seed)
        warm_start_model = None
        warm_start_normalizer = None
        previous_stage = None
        for stage in stages:
            stage_num_envs = num_envs_override if num_envs_override is not None else stage.num_envs
            warm_start_model, warm_start_normalizer = train_stage(
                stage=stage,
                seed=seed,
                artifact_root=artifact_root,
                num_envs=stage_num_envs,
                warm_start_model=warm_start_model,
                warm_start_normalizer=warm_start_normalizer,
                previous_stage=previous_stage,
                ppo_kwargs=ppo_kwargs,
                policy_kwargs=policy_kwargs,
                feature_extractor_kwargs=feature_extractor_kwargs,
            )
            previous_stage = stage