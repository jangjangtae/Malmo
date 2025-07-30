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
            #"move -1",
            "turn 1",
            "turn -1",
            #"move 1;turn 1",
            #"move 1;turn -1",
            #"jump 1",
            "move 0",
            #"pitch 1",       # 위를 바라보게
            #"pitch -1",      # 아래를 바라보게
            #"hotbar.1 1",    # 슬롯 1번 선택
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
            #print(f"📍 Position: x={obs['XPos']:.1f}, y={obs['YPos']:.1f}, z={obs['ZPos']:.1f}")
            return np.array([x, z, yaw], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def step(self, action_idx):
      action = self.action_list[action_idx]
      #print(f"👉 Executing action: {action}")

      if not self.agent_host.getWorldState().is_mission_running:
        print("❌ Mission ended early.")
        return np.zeros(3, dtype=np.float32), 0, True, False, {}

      # 명령 실행
      for cmd in action.split(";"):
        self.agent_host.sendCommand(cmd)
      time.sleep(0.4)

      obs = self._get_observation()
      reward = 0
      world_state = self.agent_host.getWorldState()

      # 1️⃣ 기본 보상(다이아몬드 블록 접근)
      if np.linalg.norm(obs[:2] - np.array([3, 3])) < 1.5:
        reward += 1

      # 2️⃣ 강제 버그 발생 시뮬레이션
      if "use 1" in action:
        import random
        if random.random() < 0.5:
            print("[BUG DETECTED] Door did not open after interaction")
            reward += 10

      # 3️⃣ 근처에 아이템 있으면 소량 보상
      if np.linalg.norm(obs[:2] - np.array([4, 4])) < 1.5:
        reward += 1

      terminated = False
      truncated = False

      print(f"🎯 Step reward: {reward}")
      self.episode_reward += reward

      return obs, reward, terminated, truncated, {}

    #def _compute_reward(self, obs):
    #  world_state = self.agent_host.getWorldState()
    #  total_reward = 0
    #  for r in world_state.rewards:
    #    total_reward += r.getValue()
    #  return total_reward

    def _check_bug_reward(self, world_state):
      reward = 0
      messages = []

      # 1) Malmo 에러 로그 읽기
      for error in world_state.errors:
        print(f"🚨 [MALMO ERROR]: {error.text}")   # 🔹 에러 로그 출력
        messages.append(error.text)

      # 2) 채팅/관측 메시지 읽기
      for obs in world_state.observations:
        if hasattr(obs, "text"):
            messages.append(obs.text)
        elif isinstance(obs, dict) and "text" in obs:
            messages.append(obs["text"])

      # 3) 버그 정의와 매칭
      for bug in self.known_bugs:
        if any(bug["message"] in msg for msg in messages):
            if bug["id"] not in self.detected_bugs:
                self.detected_bugs.add(bug["id"])
                reward += 10
                print(f"🐞 [BUG DETECTED] {bug['id']}: {bug['message']} -> +10 reward")

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
