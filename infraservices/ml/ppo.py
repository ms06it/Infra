import numpy as np
import gym

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    class HealingEnv(gym.Env):
        def __init__(self):
            super(HealingEnv, self).__init__()
            self.action_space = gym.spaces.Discrete(2)  # 0: do nothing, 1: restart service
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
            self.state = np.random.rand(4)
            self.steps = 0

        def reset(self):
            self.state = np.random.rand(4)
            self.steps = 0
            return self.state

        def step(self, action):
            self.steps += 1
            reward = 1 if action == 1 and self.state[3] > 0.7 else -1
            done = self.steps >= 10
            self.state = np.random.rand(4)
            return self.state, reward, done, {}

    env = make_vec_env(HealingEnv, n_envs=1)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_healing_model")
    print("✅ PPO model trained successfully.")

except ImportError:
    print("⚠️ PPO requires 'stable-baselines3'. Install it with: pip install stable-baselines3[extra]")

