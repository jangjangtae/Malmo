from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.malmo_env import MalmoEnv

env = MalmoEnv()

model = PPO("MlpPolicy", env, verbose=1)
print("학습 시작")
model.learn(total_timesteps=1000)
print("학습 끝")
model.save("ppo_malmo_explorer")
