# malmo_env.py

import gym
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import MalmoPython
import time
import random
import json

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
            "pitch 1",       # 위를 바라보게
            "pitch -1",      # 아래를 바라보게
            "hotbar.1 1",    # 슬롯 1번 선택
            "use 1"          # 블록 설치 시도
        ]
        self.action_space = spaces.Discrete(len(self.action_list))
        # 위치(x, z), yaw (방향)
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(3,), dtype=np.float32)
        with open("envs/bug_definitions.json", "r") as f:
            self.known_bugs = json.load(f)["bugs"]
        self.detected_bugs = set()

    def _get_mission_xml(self):
      with open("missions/bug_mission.xml", "r") as f:
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
      self.episode_reward = 0
      self.detected_bugs = set()
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
            print(f"📍 Position: x={obs['XPos']:.1f}, y={obs['YPos']:.1f}, z={obs['ZPos']:.1f}")
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

      # 명령 실행
      for cmd in action.split(";"):
        self.agent_host.sendCommand(cmd)
      time.sleep(0.4)

      # 최신 상태 수집
      obs = self._get_observation()

      # ✅ XML 정의 기반 보상 수집 (diamond_block 보상 등)
      reward = 0
      world_state = self.agent_host.getWorldState()
      for r in world_state.rewards:
        reward += r.getValue()

      # 종료 조건 확인
      terminated = self._check_done(obs)
      truncated = False

      # 로깅 및 누적
      print(f"🎯 Step reward: {reward}")
      self.episode_reward += reward
      if terminated or truncated:
        print(f"🏁 Episode done! Total reward: {self.episode_reward}")

      return obs, reward, terminated, truncated, {}

    #def _compute_reward(self, obs):
    #  world_state = self.agent_host.getWorldState()
    #  total_reward = 0
    #  for r in world_state.rewards:
    #    total_reward += r.getValue()
    #  return total_reward

    def check_bug_reward(world_state):
      reward = 0
      messages = []

      # 1) Malmo에서 에러 메시지 읽기
      for error in world_state.errors:
        messages.append(error.text)
      for obs in world_state.observations:
        if "text" in obs:
            messages.append(obs["text"])

      # 2) 버그 정의와 매칭
      for bug in bug_data["bugs"]:
        if any(bug["message"] in msg for msg in messages):
            reward += 10  # 버그 탐지 보상
            print(f"[BUG DETECTED] {bug['id']}: {bug['message']} -> +10 reward")

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
