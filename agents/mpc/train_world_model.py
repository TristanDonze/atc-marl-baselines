import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import aim
from tqdm import tqdm
from src.networks import LearnedDynamics, LearnedCost

def train_world_model():
    run = aim.Run(experiment="world_model_training")

    print("Loading dataset...")
    dataset = torch.load("dataset/world_model_data.pt")
    
    states = dataset["states"]
    actions = dataset["actions"]
    next_states = dataset["next_states"]
    costs = -dataset["rewards"]

    dataset_size = len(states)
    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size

    print(f"Dataset loaded. Train size: {train_size}, Val size: {val_size}")

    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_states = states[train_indices]
    train_actions = actions[train_indices]
    train_next_states = next_states[train_indices]
    train_costs = costs[train_indices]

    val_states = states[val_indices]
    val_actions = actions[val_indices]
    val_next_states = next_states[val_indices]
    val_costs = costs[val_indices]

    print("Calculating normalization parameters...")
    state_mean = train_states.mean(dim=0, keepdim=True)
    state_std = train_states.std(dim=0, keepdim=True) + 1e-8
    cost_mean = train_costs.mean(dim=0, keepdim=True)
    cost_std = train_costs.std(dim=0, keepdim=True) + 1e-8

    norm_train_states = (train_states - state_mean) / state_std
    norm_train_next_states = (train_next_states - state_mean) / state_std
    norm_train_costs = (train_costs - cost_mean) / cost_std

    norm_val_states = (val_states - state_mean) / state_std
    norm_val_next_states = (val_next_states - state_mean) / state_std
    norm_val_costs = (val_costs - cost_mean) / cost_std

    norm_train_states = norm_train_states.cuda()
    train_actions = train_actions.cuda()
    norm_train_next_states = norm_train_next_states.cuda()
    norm_train_costs = norm_train_costs.cuda()

    norm_val_states = norm_val_states.cuda()
    val_actions = val_actions.cuda()
    norm_val_next_states = norm_val_next_states.cuda()
    norm_val_costs = norm_val_costs.cuda()

    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    batch_size = 2048
    epochs = 50
    learning_rate = 1e-3

    run["hparams"] = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_size": train_size,
        "val_size": val_size
    }

    dynamics_model = LearnedDynamics(state_dim, action_dim).cuda()
    cost_model = LearnedCost(state_dim, action_dim).cuda()

    dyn_optimizer = optim.AdamW(dynamics_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    cost_optimizer = optim.AdamW(cost_model.parameters(), lr=learning_rate, weight_decay=5e-2)

    dyn_scheduler = CosineAnnealingLR(dyn_optimizer, T_max=epochs)
    cost_scheduler = CosineAnnealingLR(cost_optimizer, T_max=epochs)

    mse_loss = nn.MSELoss()
    huber_loss = nn.SmoothL1Loss()

    train_data = TensorDataset(norm_train_states, train_actions, norm_train_next_states, norm_train_costs)
    val_data = TensorDataset(norm_val_states, val_actions, norm_val_next_states, norm_val_costs)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    torch.save({
        "state_mean": state_mean.cpu(),
        "state_std": state_std.cpu(),
        "cost_mean": cost_mean.cpu(),
        "cost_std": cost_std.cpu()
    }, "dataset/normalization_params.pt")

    best_val_dyn_loss = float('inf')
    best_val_cost_loss = float('inf')
    
    noise_std = 0.01

    print("Starting training loop...")
    for epoch in range(epochs):
        dynamics_model.train()
        cost_model.train()
        total_dyn_loss = 0.0
        total_cost_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_states, batch_actions, batch_next_states, batch_costs in pbar:
            dyn_optimizer.zero_grad()
            pred_next_states = dynamics_model(batch_states, batch_actions)
            dyn_loss = mse_loss(pred_next_states, batch_next_states)
            dyn_loss.backward()
            dyn_optimizer.step()

            cost_optimizer.zero_grad()
            
            tau = torch.cat([batch_states, batch_actions], dim=-1)
            tau = tau + torch.randn_like(tau) * noise_std # Add noise to improve generalization
            
            pred_costs = cost_model(tau)
            cost_loss = huber_loss(pred_costs, batch_costs.squeeze(-1))
            cost_loss.backward()
            cost_optimizer.step()

            total_dyn_loss += dyn_loss.item()
            total_cost_loss += cost_loss.item()
            
            pbar.set_postfix({
                "dyn_loss": f"{total_dyn_loss / (pbar.n + 1):.4f}",
                "cost_loss": f"{total_cost_loss / (pbar.n + 1):.4f}"
            })
            
        dyn_scheduler.step()
        cost_scheduler.step()

        avg_train_dyn_loss = total_dyn_loss / len(train_loader)
        avg_train_cost_loss = total_cost_loss / len(train_loader)

        dynamics_model.eval()
        cost_model.eval()
        val_dyn_loss = 0.0
        val_cost_loss = 0.0

        with torch.no_grad():
            for batch_states, batch_actions, batch_next_states, batch_costs in val_loader:
                pred_next_states = dynamics_model(batch_states, batch_actions)
                loss_d = mse_loss(pred_next_states, batch_next_states)
                val_dyn_loss += loss_d.item()

                tau = torch.cat([batch_states, batch_actions], dim=-1)
                pred_costs = cost_model(tau)
                loss_c = huber_loss(pred_costs, batch_costs.squeeze(-1))
                val_cost_loss += loss_c.item()

        avg_val_dyn_loss = val_dyn_loss / len(val_loader)
        avg_val_cost_loss = val_cost_loss / len(val_loader)

        run.track(avg_train_dyn_loss, name="loss", context={"subset": "train", "model": "dynamics"}, epoch=epoch)
        run.track(avg_train_cost_loss, name="loss", context={"subset": "train", "model": "cost"}, epoch=epoch)
        run.track(avg_val_dyn_loss, name="loss", context={"subset": "val", "model": "dynamics"}, epoch=epoch)
        run.track(avg_val_cost_loss, name="loss", context={"subset": "val", "model": "cost"}, epoch=epoch)

        print(f"Epoch {epoch+1}/{epochs} Summary | Val Dyn: {avg_val_dyn_loss:.4f} | Val Cost: {avg_val_cost_loss:.4f}")

        if avg_val_dyn_loss < best_val_dyn_loss:
            best_val_dyn_loss = avg_val_dyn_loss
            torch.save(dynamics_model.state_dict(), "dataset/dynamics_model.pth")
            print(f"--> Saved new best dynamics model! (Loss: {best_val_dyn_loss:.4f})")

        if avg_val_cost_loss < best_val_cost_loss:
            best_val_cost_loss = avg_val_cost_loss
            torch.save(cost_model.state_dict(), "dataset/cost_model.pth")
            print(f"--> Saved new best cost model! (Loss: {best_val_cost_loss:.4f})")

    print("Training complete.")

if __name__ == "__main__":
    train_world_model()