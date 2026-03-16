from pathlib import Path
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

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