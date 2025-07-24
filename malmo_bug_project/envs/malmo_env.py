import MalmoPython
import time
import numpy as np

class MalmoEnv:
    def __init__(self, mission_file):
        with open(mission_file, 'r') as f:
            mission_xml = f.read()
        self.agent_host = MalmoPython.AgentHost()
        self.mission = MalmoPython.MissionSpec(mission_xml, True)
        self.mission_record = MalmoPython.MissionRecordSpec()
        self.state_dim = 5  # 예시
        self.action_space = 4  # 앞, 뒤, 좌, 우

    def reset(self):
        self.agent_host.startMission(self.mission, self.mission_record)
        while not self.agent_host.getWorldState().has_mission_begun:
            time.sleep(0.1)
        return self._get_obs()

    def step(self, action):
        actions = ["move 1", "move -1", "strafe -1", "strafe 1"]
        self.agent_host.sendCommand(actions[action])
        time.sleep(0.2)

        ws = self.agent_host.getWorldState()
        reward = 0
        done = not ws.is_mission_running

        for r in ws.rewards:
            reward += r.getValue()

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # 간단히 위치정보 반환
        ws = self.agent_host.getWorldState()
        if ws.number_of_observations > 0:
            obs_text = ws.observations[-1].text
            obs = eval(obs_text)
            return np.array([
                obs.get("XPos", 0),
                obs.get("ZPos", 0),
                obs.get("Yaw", 0),
                obs.get("Health", 0),
                obs.get("Inventory", [])
            ], dtype=object)
        else:
            return np.zeros(5, dtype=object)
