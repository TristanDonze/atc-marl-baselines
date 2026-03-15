import torch
from torch import nn

class AirTrafficDx(nn.Module):
    def __init__(self, max_planes, enable_accel, enable_wind):
        super().__init__()
        self.max_planes = max_planes
        self.enable_accel = enable_accel
        self.enable_wind = enable_wind
        
        self.turn_rate = 0.05
        self.accel_rate = 0.1
        self.min_speed = 1.0
        self.max_speed = 5.0

    def forward(self, state, u):
        squeeze = state.ndimension() == 1
        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)

        batch_size = state.shape[0]

        if self.enable_wind:
            wind_x = state[:, -2].unsqueeze(1)  # Shape: [batch_size, 1]
            wind_y = state[:, -1].unsqueeze(1)
        else:
            wind_x = 0.0
            wind_y = 0.0

        # p_state shape: [batch_size, max_planes, 6]
        p_state = state[:, :self.max_planes * 6].view(batch_size, self.max_planes, 6)
        
        action_dim = 2 if self.enable_accel else 1
        u_reshaped = u.view(batch_size, self.max_planes, action_dim)

        x = p_state[:, :, 0]
        y = p_state[:, :, 1]
        speed = p_state[:, :, 2]
        heading = p_state[:, :, 3]
        active = p_state[:, :, 4]
        dest_id = p_state[:, :, 5]

        steering = torch.clamp(u_reshaped[:, :, 0], -1.0, 1.0)
        new_heading = heading + (steering * self.turn_rate)

        if self.enable_accel:
            throttle = torch.clamp(u_reshaped[:, :, 1], -1.0, 1.0)
            new_speed = torch.clamp(speed + (throttle * self.accel_rate), self.min_speed, self.max_speed)
        else:
            new_speed = speed

        dx = new_speed * torch.cos(new_heading) + wind_x
        dy = new_speed * torch.sin(new_heading) + wind_y

        new_x = x + dx
        new_y = y + dy

        new_p_state = torch.stack([new_x, new_y, new_speed, new_heading, active, dest_id], dim=2)
        new_state_flat = new_p_state.view(batch_size, self.max_planes * 6)

        if self.enable_wind:
            new_state = torch.cat([new_state_flat, state[:, -2:]], dim=1)
        else:
            new_state = new_state_flat

        if squeeze:
            new_state = new_state.squeeze(0)

        return new_state


class AirTrafficCost(nn.Module):
    def __init__(self, max_planes, zones_dict, enable_wind):
        super().__init__()
        self.max_planes = max_planes
        self.zones = zones_dict
        self.enable_wind = enable_wind
        self.state_dim = self.max_planes * 6 + (2 if self.enable_wind else 0)

    def forward(self, tau):
        batch_size = tau.shape[0]
        device = tau.device
        cost = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        actions = tau[:, self.state_dim:]
        cost = cost + torch.sum(actions**2, dim=1) * 0.1

        p_state = tau[:, :self.max_planes * 6].view(batch_size, self.max_planes, 6)
        
        x = p_state[:, :, 0] / 800.0
        y = p_state[:, :, 1] / 600.0
        heading = p_state[:, :, 3]
        active = p_state[:, :, 4].detach()
        dest_id = p_state[:, :, 5].detach()

        for z_id, z_data in self.zones.items():
            zx, zy, zangle, is_heli = z_data
            zx_norm = zx / 800.0
            zy_norm = zy / 600.0
            
            target_mask = (dest_id == float(z_id)).float() * active
            
            if is_heli == 1.0:
                rabbit_x = zx_norm
                rabbit_y = zy_norm
            else:
                rabbit_x = torch.clamp(x + 0.15, max=zx_norm)
                rabbit_y = zy_norm

            dist_sq = (x - rabbit_x)**2 + (y - rabbit_y)**2
            cost = cost + torch.sum(dist_sq * target_mask * 40.0, dim=1)

            ideal_heading = torch.atan2(rabbit_y - y, rabbit_x - x)
            h_diff = heading - ideal_heading
            h_diff = torch.atan2(torch.sin(h_diff), torch.cos(h_diff))
            
            cost = cost + torch.sum((h_diff ** 2) * target_mask * 15.0, dim=1)
            
            if is_heli == 0.0:
                final_h_diff = heading - zangle
                final_h_diff = torch.atan2(torch.sin(final_h_diff), torch.cos(final_h_diff))
                
                proximity_weight = torch.exp(-15.0 * dist_sq)
                cost = cost + torch.sum((final_h_diff ** 2) * proximity_weight * target_mask * 20.0, dim=1)

        if self.max_planes > 1:
            dx = x.unsqueeze(2) - x.unsqueeze(1)
            dy = y.unsqueeze(2) - y.unsqueeze(1)
            
            dist_matrix_sq = dx**2 + dy**2
            active_matrix = active.unsqueeze(2) * active.unsqueeze(1)
            
            triu_mask = torch.triu(torch.ones(self.max_planes, self.max_planes, device=device), diagonal=1)
            
            repulsion = 0.5 / (dist_matrix_sq + 0.001)
            cost = cost + torch.sum(repulsion * active_matrix * triu_mask, dim=(1, 2))

        return cost