# malmo_env.py

import gym
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import MalmoPython
import time
import random

class MalmoEnv(gym.Env):
    def __init__(self):
        super(MalmoEnv, self).__init__()
        self.agent_host = MalmoPython.AgentHost()
        self.mission_xml = self._get_mission_xml()
        self.action_list = [
            "move 1",
            "move -1",
            "turn 1",
            "turn -1",
            "move 1;turn 1",
            "move 1;turn -1",
            "jump 1",
            "move 0"
        ]
        self.action_space = spaces.Discrete(len(self.action_list))
        # 위치(x, z), yaw (방향)
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(3,), dtype=np.float32)

    def _get_mission_xml(self):
      with open("missions/block_bug.xml", "r") as f:
        return f.read()

    def reset(self, *, seed=None, options=None):
      if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

      mission_spec = MalmoPython.MissionSpec(self.mission_xml, True)
      mission_record = MalmoPython.MissionRecordSpec()

      # 종료된 미션 정리 대기
      while self.agent_host.getWorldState().is_mission_running:
        time.sleep(0.5)

      # 새 미션 시작
      for attempt in range(5):
        try:
            self.agent_host.startMission(mission_spec, mission_record)
            break
        except RuntimeError as e:
            print(f"❗ Mission start failed: {e}")
            time.sleep(1)

      # 미션 시작 대기
      while True:
        world_state = self.agent_host.getWorldState()
        if world_state.has_mission_begun:
            break
        time.sleep(0.2)

      time.sleep(0.5)
      obs = self._get_observation()
      return obs, {}


    def _get_observation(self):
        world_state = self.agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            obs_text = world_state.observations[-1].text
            import json
            obs = json.loads(obs_text)
            x = obs.get("XPos", 0)
            z = obs.get("ZPos", 0)
            yaw = obs.get("Yaw", 0)
            return np.array([x, z, yaw], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def step(self, action_idx):
      action = self.action_list[action_idx]
      print(f"👉 Executing action: {action}")
      world_state = self.agent_host.getWorldState()
      if not world_state.is_mission_running:
        print("❌ Mission has ended — force terminating episode.")
        obs = self._get_observation()
        reward = 0
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, {}

      for cmd in action.split(";"):
        self.agent_host.sendCommand(cmd)
      time.sleep(0.2)

      obs = self._get_observation()
      reward = self._compute_reward(obs)
      terminated = self._check_done(obs)
      truncated = False  # 또는 step count > max_steps 등으로 처리

      info = {}  # 필수 (stable-baselines3 내부에서 사용)

      return obs, reward, terminated, truncated, info

    def _compute_reward(self, obs):
        # 예시: 목표 지점(5,5)에 가까워지면 +보상
        target = np.array([5, 5])
        pos = np.array([obs[0], obs[1]])
        dist = np.linalg.norm(pos - target)
        reward = -dist / 20  # 거리가 가까울수록 높음
        if dist < 1.0:
            reward += 10  # 목표 도달 시 큰 보상
        return reward

    def _check_done(self, obs):
        pos = np.array([obs[0], obs[1]])
        if np.linalg.norm(pos - np.array([5, 5])) < 1.0:
            return True
        return False

    def render(self, mode="human"):
        pass

    def close(self):
        pass
