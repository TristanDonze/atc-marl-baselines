import os
import supersuit as ss
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecEnvWrapper
from stable_baselines3.common import results_plotter
from gym_air_traffic.envs.air_traffic_env import AirTrafficEnv

# --- FIX: Wrapper to convert integer dones to booleans for VecNormalize ---
class BooleanDoneWrapper(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        dones = np.asarray(dones, dtype=np.bool_)
        return obs, rewards, dones, infos
# ------------------------------------------------------------------------

best_mean_reward = -np.inf

def callback(_locals, _globals):
    global best_mean_reward
    model_instance = _locals['self']
    
    if len(model_instance.ep_info_buffer) > 0:
        mean_reward = np.mean([ep_info['r'] for ep_info in model_instance.ep_info_buffer])
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            model_instance.save("models/ppo_air_traffic_best")
            
            # Save the normalization stats for the best model
            env = model_instance.get_env()
            if env is not None:
                env.save("models/vec_normalize_best.pkl")
                
            print(f"New best model saved with mean reward: {best_mean_reward:.2f}")
    return True

def main():
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)

    # --- TRAINING ENVIRONMENT SETUP ---
    env = AirTrafficEnv(render_mode=None, max_planes=1, enable_acceleration=False, enable_wind=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    
    # Apply our custom fix and SB3 wrappers
    env = BooleanDoneWrapper(env)
    env = VecMonitor(env, log_dir)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        tensorboard_log=log_dir,
        batch_size=256,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128])
    )

    learn = False # Set to True to train, False to just evaluate
    if learn:
        print("Starting training...")
        os.makedirs("models", exist_ok=True)
        
        model.learn(total_timesteps=3_000_000, tb_log_name="ppo_air_traffic", callback=callback)
        model.save("models/ppo_air_traffic")
        env.save("models/vec_normalize.pkl") # Save final stats
        print("Training finished and model saved.")

        results_plotter.plot_results([log_dir], 500_000, results_plotter.X_TIMESTEPS, "PPO Training Reward", figsize=(20, 12))
        plt.savefig("models/reward_plot.png")
    
    # --- EVALUATION SETUP ---
    model = PPO.load("models/ppo_air_traffic")
    
    # Dummy env to load the normalizer
    dummy_env = AirTrafficEnv(render_mode=None, max_planes=1, enable_acceleration=False, enable_wind=False)
    dummy_env = ss.pettingzoo_env_to_vec_env_v1(dummy_env)
    dummy_env = ss.concat_vec_envs_v1(dummy_env, 1, num_cpus=1, base_class="stable_baselines3")
    dummy_env = BooleanDoneWrapper(dummy_env)
    
    vec_normalize = VecNormalize.load("models/vec_normalize.pkl", dummy_env)
    vec_normalize.training = False     # Do not update stats during eval
    vec_normalize.norm_reward = False  # Only normalize obs during eval

    test_env = AirTrafficEnv(render_mode="rgb_array", max_planes=1, enable_acceleration=False, enable_wind=False)
    frames = []
    observations, infos = test_env.reset()

    print("Starting evaluation...")
    try:
        while test_env.steps < 1000:
            actions = {}
            for agent in test_env.agents:
                obs = observations[agent]
                if obs[0] == -1.0:
                    actions[agent] = np.zeros(test_env.action_dim, dtype=np.float32)
                    continue

                obs_batched = np.expand_dims(obs, axis=0)
                obs_batched = vec_normalize.normalize_obs(obs_batched) # Normalize the test state
                
                action, _states = model.predict(obs_batched, deterministic=True)
                actions[agent] = action[0]

            observations, rewards, terminations, truncations, infos = test_env.step(actions)
            frame = test_env.render()
            if frame is not None:
                frames.append(frame)

            if all(terminations.values()) or all(truncations.values()):
                print(f"Episode finished at step {test_env.steps}")
                break

    except KeyboardInterrupt:
        print("Evaluation stopped.")

    test_env.save_video("videos", frames, filename="ppo_eval.mp4", fps=30)
    test_env.close()

if __name__ == "__main__":
    main()