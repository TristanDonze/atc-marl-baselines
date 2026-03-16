import torch
import torch.nn as nn

class StructuredAirTrafficExtractor(nn.Module):
    def __init__(self, self_feature_dim, neighbor_feature_dim, max_neighbors, features_dim=384, self_hidden_dim=192, neighbor_hidden_dim=192):
        super().__init__()
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

    def forward(self, observations):
        self_obs = observations[:, :self.self_feature_dim]
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
            neighbor_obs = observations[:, self.self_feature_dim:]
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

class AirTrafficActorCriticNetwork(nn.Module):
    def __init__(self, self_feature_dim, neighbor_feature_dim, max_neighbors, n_actions, action_space_type="continuous"):
        super().__init__()
        self.action_space_type = action_space_type
        
        self.extractor = StructuredAirTrafficExtractor(
            self_feature_dim=self_feature_dim,
            neighbor_feature_dim=neighbor_feature_dim,
            max_neighbors=max_neighbors
        )
        
        self.actor = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        
        if self.action_space_type == "continuous":
            self.action_log_std = nn.Parameter(torch.zeros(1, n_actions))
        
        self.critic = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def get_action_distribution_params(self, state):
        features = self.extractor(state)
        action_mean_or_logits = self.actor(features)
        
        if self.action_space_type == "continuous":
            action_log_std = self.action_log_std.expand_as(action_mean_or_logits)
            return action_mean_or_logits, action_log_std
        
        return action_mean_or_logits, None

    def get_state_value(self, state):
        features = self.extractor(state)
        return self.critic(features)