from dataclasses import dataclass

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