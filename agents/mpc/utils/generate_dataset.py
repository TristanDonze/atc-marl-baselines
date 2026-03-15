import os
import torch
import numpy as np
from multiprocessing import Pool, cpu_count
from gym_air_traffic.envs.air_traffic_env import AirTrafficEnv
from tqdm import tqdm

def collect_chunk(args):
    chunk_size, max_planes, enable_accel, enable_wind, seed, pbar_pos = args
    
    env = AirTrafficEnv(
        render_mode=None,
        max_planes=max_planes,
        enable_acceleration=enable_accel,
        enable_wind=enable_wind
    )
    env.reset(seed=seed)
    
    states = []
    actions = []
    next_states = []
    rewards = []
    
    current_state = env.get_mpc_state()
    collected = 0

    with tqdm(total=chunk_size, position=pbar_pos, desc=f"Worker {pbar_pos}", leave=False) as pbar:
        while collected < chunk_size:
            action_dict = {
                agent: env.action_space(agent).sample() 
                for agent in env.agents 
                if env.planes_dict[agent] is not None and env.planes_dict[agent].active
            }
            
            _, step_rewards, term, trunc, _ = env.step(action_dict)
            next_state = env.get_mpc_state()
            
            flat_action = []
            for agent in env.possible_agents:
                if agent in action_dict:
                    flat_action.extend(action_dict[agent].tolist())
                else:
                    flat_action.extend([0.0] * (2 if enable_accel else 1))
                    
            total_reward = sum(step_rewards.values())
            
            states.append(current_state)
            actions.append(flat_action)
            next_states.append(next_state)
            rewards.append(total_reward)
            
            collected += 1
            pbar.update(1)
            
            if all(term.values()) or all(trunc.values()):
                env.reset()
                current_state = env.get_mpc_state()
            else:
                current_state = next_state
                
    env.close()
    return states, actions, next_states, rewards

def generate_data_vectorized(total_samples, max_planes, enable_accel, enable_wind):
    num_workers = cpu_count()
    chunk_size = total_samples // num_workers
    
    print(f"Starting {num_workers} workers to collect {chunk_size} samples each")
    
    args_list = [
        (chunk_size, max_planes, enable_accel, enable_wind, i, i)
        for i in range(num_workers)
    ]
    
    with Pool(num_workers) as pool:
        results = pool.map(collect_chunk, args_list)
        
    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    
    for s, a, ns, r in results:
        all_states.extend(s)
        all_actions.extend(a)
        all_next_states.extend(ns)
        all_rewards.extend(r)
        
    os.makedirs("dataset", exist_ok=True)
    
    torch.save({
        "states": torch.tensor(np.array(all_states), dtype=torch.float32),
        "actions": torch.tensor(np.array(all_actions), dtype=torch.float32),
        "next_states": torch.tensor(np.array(all_next_states), dtype=torch.float32),
        "rewards": torch.tensor(np.array(all_rewards), dtype=torch.float32).unsqueeze(1)
    }, "dataset/world_model_data.pt")
    
    print("Dataset generated successfully")

if __name__ == "__main__":
    generate_data_vectorized(5_000_000, 10, True, True)