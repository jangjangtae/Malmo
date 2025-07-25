import MalmoPython
import time
import numpy as np
import json

class MalmoEnv:
    def __init__(self, mission_file):
        with open(mission_file, 'r') as f:
            mission_xml = f.read()
        self.agent_host = MalmoPython.AgentHost()
        self.mission = MalmoPython.MissionSpec(mission_xml, True)
        self.mission_record = MalmoPython.MissionRecordSpec()
        self.state_dim = 5
        self.action_space = 5  # 개별 행동 단위 수

        self.actions = ['move 1', 'move -1', 'turn -1', 'turn 1', 'jump 1']

    def reset(self):
        for retry in range(3):
            try:
                self.agent_host.startMission(self.mission, self.mission_record)
                break
            except RuntimeError as e:
                if retry == 2:
                    raise e
                print(f"[MalmoEnv] Mission start failed (retry {retry+1}), waiting...")
                time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("[MalmoEnv] Error:", error.text)

        return self._get_obs()

    def step(self, action_vector):
        assert len(action_vector) == self.action_space

        # 매 스텝마다 선택된 여러 행동을 조합 실행
        for i, act in enumerate(self.actions):
            if action_vector[i]:
                self.agent_host.sendCommand(act)
                print(act)  # 실행된 동작 로그

        time.sleep(0.2)

        ws = self.agent_host.getWorldState()
        while len(ws.rewards) == 0 and ws.is_mission_running:
            time.sleep(0.1)
            ws = self.agent_host.getWorldState()

        reward = sum(r.getValue() for r in ws.rewards)
        done = not ws.is_mission_running

        if reward >= 10:
            print("[✓] Bug block stepped on! Reward received.")

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        ws = self.agent_host.getWorldState()
        while ws.is_mission_running and len(ws.observations) == 0:
            time.sleep(0.05)
            ws = self.agent_host.getWorldState()

        if len(ws.observations) == 0:
            return np.zeros(5, dtype=np.float32)

        obs_text = ws.observations[-1].text
        obs = json.loads(obs_text)

        x = float(obs.get("XPos", 0))
        y = float(obs.get("YPos", 0))
        z = float(obs.get("ZPos", 0))
        yaw = float(obs.get("Yaw", 0))
        pitch = float(obs.get("Pitch", 0))

        return np.array([x, y, z, yaw, pitch], dtype=np.float32)
