import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class StructuredAirTrafficExtractor(BaseFeaturesExtractor):
    """Encode self state separately from repeated neighbor blocks."""

    def __init__(
        self,
        observation_space: spaces.Box,
        self_feature_dim: int,
        neighbor_feature_dim: int,
        max_neighbors: int,
        features_dim: int = 384,
        self_hidden_dim: int = 192,
        neighbor_hidden_dim: int = 192,
    ) -> None:
        super().__init__(observation_space, features_dim)
        expected_obs_dim = self_feature_dim + (neighbor_feature_dim * max_neighbors)
        actual_obs_dim = int(observation_space.shape[0])
        if actual_obs_dim != expected_obs_dim:
            raise ValueError(
                "StructuredAirTrafficExtractor received inconsistent observation dimensions: "
                f"expected {expected_obs_dim}, got {actual_obs_dim}"
            )

        self.self_feature_dim = self_feature_dim
        self.neighbor_feature_dim = neighbor_feature_dim
        self.max_neighbors = max_neighbors
        self.neighbor_hidden_dim = neighbor_hidden_dim

        self.self_encoder = nn.Sequential(
            nn.Linear(self_feature_dim, self_hidden_dim),
            nn.ReLU(),
            nn.Linear(self_hidden_dim, self_hidden_dim),
            nn.ReLU(),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_feature_dim, neighbor_hidden_dim),
            nn.ReLU(),
            nn.Linear(neighbor_hidden_dim, neighbor_hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self_hidden_dim + (2 * neighbor_hidden_dim) + 1, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self_obs = observations[:, : self.self_feature_dim]
        self_embedding = self.self_encoder(self_obs)

        if self.max_neighbors == 0:
            neighbor_mean = torch.zeros(
                (observations.shape[0], self.neighbor_hidden_dim),
                device=observations.device,
                dtype=observations.dtype,
            )
            neighbor_max = torch.zeros_like(neighbor_mean)
            neighbor_count = torch.zeros(
                (observations.shape[0], 1),
                device=observations.device,
                dtype=observations.dtype,
            )
        else:
            neighbor_obs = observations[:, self.self_feature_dim :]
            neighbor_obs = neighbor_obs.reshape(-1, self.max_neighbors, self.neighbor_feature_dim)
            neighbor_mask = (neighbor_obs[:, :, -1] > 0.0).to(observations.dtype).unsqueeze(-1)

            neighbor_embeddings = self.neighbor_encoder(neighbor_obs)
            masked_neighbor_embeddings = neighbor_embeddings * neighbor_mask

            neighbor_count = neighbor_mask.sum(dim=1).clamp(min=1.0)
            neighbor_mean = masked_neighbor_embeddings.sum(dim=1) / neighbor_count

            masked_for_max = neighbor_embeddings.masked_fill(neighbor_mask == 0.0, float("-inf"))
            neighbor_max = masked_for_max.max(dim=1).values
            has_neighbor = (neighbor_mask.sum(dim=1) > 0.0).expand_as(neighbor_max)
            neighbor_max = torch.where(has_neighbor, neighbor_max, torch.zeros_like(neighbor_max))
            neighbor_count = neighbor_count / max(1, self.max_neighbors)

        fused = torch.cat([self_embedding, neighbor_mean, neighbor_max, neighbor_count], dim=1)
        return self.fusion(fused)
