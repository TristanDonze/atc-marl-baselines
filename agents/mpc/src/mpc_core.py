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
        # DEBUG PRINT
        if torch.isnan(state).any() or torch.isnan(u).any():
            print(f"DEBUG Dx - INPUT IS NaN! state: {state}, u: {u}")

        squeeze = state.ndimension() == 1
        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)

        new_state_elements = []

        for i in range(self.max_planes):
            idx = i * 6
            x = state[:, idx]
            y = state[:, idx + 1]
            speed = state[:, idx + 2]
            heading = state[:, idx + 3]
            active = state[:, idx + 4]
            dest_id = state[:, idx + 5]

            u_idx = i * 2 if self.enable_accel else i
            steering = torch.clamp(u[:, u_idx], -1.0, 1.0)
            
            new_heading = heading + (steering * self.turn_rate)

            if self.enable_accel:
                throttle = torch.clamp(u[:, u_idx + 1], -1.0, 1.0)
                new_speed = torch.clamp(speed + (throttle * self.accel_rate), self.min_speed, self.max_speed)
            else:
                new_speed = speed

            wind_x = state[:, -2] if self.enable_wind else 0.0
            wind_y = state[:, -1] if self.enable_wind else 0.0

            dx = new_speed * torch.cos(new_heading) + wind_x
            dy = new_speed * torch.sin(new_heading) + wind_y

            new_x = x + dx
            new_y = y + dy

            new_state_elements.extend([new_x, new_y, new_speed, new_heading, active, dest_id])

        if self.enable_wind:
            new_state_elements.extend([state[:, -2], state[:, -1]])

        new_state = torch.stack(new_state_elements, dim=1)

        if squeeze:
            new_state = new_state.squeeze(0)

        # DEBUG PRINT
        if torch.isnan(new_state).any():
            print(f"DEBUG Dx - OUTPUT IS NaN! new_state: {new_state}")

        return new_state
    
class AirTrafficCost(nn.Module):
    def __init__(self, max_planes, zones_dict, enable_wind):
        super().__init__()
        self.max_planes = max_planes
        self.zones = zones_dict
        self.enable_wind = enable_wind
        self.state_dim = self.max_planes * 6 + (2 if self.enable_wind else 0)

    def forward(self, tau):
        cost = torch.zeros(tau.shape[0], dtype=torch.float32, device=tau.device)
        
        # 1. Control Penalty (Prevents NaN crashes)
        actions = tau[:, self.state_dim:]
        cost = cost + torch.sum(actions**2, dim=1) * 0.1

        for i in range(self.max_planes):
            idx = i * 6
            x = tau[:, idx] / 800.0
            y = tau[:, idx + 1] / 600.0
            heading = tau[:, idx + 3]
            
            active = tau[:, idx + 4].detach()
            dest_id = tau[:, idx + 5].detach()

            for z_id, z_data in self.zones.items():
                zx, zy, zangle, is_heli = z_data
                
                zx_norm = zx / 800.0
                zy_norm = zy / 600.0
                
                target_mask = (dest_id == float(z_id)).float() * active
                
                # 2. Distance Cost (Pull the plane to the target)
                dist_sq = (x - zx_norm)**2 + (y - zy_norm)**2
                cost = cost + (dist_sq * target_mask * 50.0) # Increased weight

                # 3. Dynamic Heading Cost (Point AT the runway, not parallel to it)
                if is_heli == 0.0:
                    ideal_heading = torch.atan2(zy_norm - y, zx_norm - x)
                    h_diff = heading - ideal_heading
                    h_diff = torch.atan2(torch.sin(h_diff), torch.cos(h_diff))
                    
                    cost = cost + ((h_diff ** 2) * target_mask * 10.0)

            # 4. Collision Repulsion
            for j in range(i + 1, self.max_planes):
                idx_j = j * 6
                xj = tau[:, idx_j] / 800.0
                yj = tau[:, idx_j + 1] / 600.0
                active_j = tau[:, idx_j + 4].detach()

                pair_active = active * active_j
                dist_sq = (x - xj)**2 + (y - yj)**2
                
                repulsion = 0.5 / (dist_sq + 0.001)
                cost = cost + (repulsion * pair_active)

        return cost