import torch
import numpy as np
from mpc import mpc
from gym_air_traffic.envs.air_traffic_env import AirTrafficEnv
from src.mpc_core import AirTrafficDx, AirTrafficCost

def main():
    max_planes = 1
    enable_accel = False
    enable_wind = False

    env = AirTrafficEnv(
        render_mode="rgb_array", 
        max_planes=max_planes, 
        enable_acceleration=enable_accel, 
        enable_wind=enable_wind
    )
    env.reset()

    nx = max_planes * 6 + (2 if enable_wind else 0)
    nu = max_planes * (2 if enable_accel else 1)
    t_horizon = 15

    dx = AirTrafficDx(max_planes, enable_accel, enable_wind)
    zones = env.get_mpc_zones()
    
    cost_fn = AirTrafficCost(max_planes, zones, enable_wind)

    ctrl = mpc.MPC(
        n_state=nx,
        n_ctrl=nu,
        T=t_horizon,
        u_lower=-1.0,
        u_upper=1.0,
        lqr_iter=5,
        exit_unconverged=False,
        eps=1e-2,
        n_batch=1,
        backprop=False,
        grad_method=mpc.GradMethods.AUTO_DIFF
    )

    frames = []
    
    for step in range(300):
        raw_state = env.get_mpc_state()
        
        # DEBUG PRINT: Verify Gym outputs
        print(f"DEBUG run_mpc - Step {step} raw_state: {raw_state}")
        
        state_tensor = torch.tensor(raw_state, dtype=torch.float32).unsqueeze(0)

        # We will wrap the solver call to catch errors
        try:
            nominal_states, nominal_actions, nominal_objs = ctrl(state_tensor, cost_fn, dx)
            optimal_action = nominal_actions[0].detach().numpy()[0]
        except Exception as e:
            print(f"DEBUG run_mpc - PyTorch/MPC threw an exception: {e}")
            optimal_action = np.zeros(nu, dtype=np.float32)
        
        if np.any(np.isnan(optimal_action)):
            print(f"Solver instability detected at step {step}: Action is NaN. Defaulting to 0.0.")
            optimal_action = np.zeros_like(optimal_action)
        else:
            print(f"DEBUG run_mpc - Step {step} Output Action: {optimal_action}")
            
        actions = {}
        for i, agent in enumerate(env.agents):
            if raw_state[i * 6 + 4] == 1.0:
                action_idx = i * 2 if enable_accel else i
                if enable_accel:
                    actions[agent] = np.array([optimal_action[action_idx], optimal_action[action_idx + 1]], dtype=np.float32)
                else:
                    actions[agent] = np.array([optimal_action[action_idx]], dtype=np.float32)

        obs, rewards, term, trunc, infos = env.step(actions)
        
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if all(term.values()) or all(trunc.values()):
            print(f"Episode finished at step {step}")
            break

    env.save_video("videos", frames, filename="mpc_test.mp4", fps=30)
    env.close()

if __name__ == "__main__":
    main()